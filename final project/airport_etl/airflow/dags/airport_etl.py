from datetime import datetime
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator

with DAG(
    dag_id="airport_etl",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",   # можно "None", если хочешь запускать вручную
    catchup=False,
    tags=["etl", "airport"],
):

    # 1. Создание схем
    create_schemas = PostgresOperator(
        task_id="create_schemas",
        postgres_conn_id="pg_dwh",
        sql="""
        CREATE SCHEMA IF NOT EXISTS ods;
        CREATE SCHEMA IF NOT EXISTS datamart;
        CREATE SCHEMA IF NOT EXISTS src_fdw;
        """,
    )

    # 2. FDW подключение
    fdw_setup = PostgresOperator(
        task_id="fdw_setup",
        postgres_conn_id="pg_dwh",
        sql="""
        CREATE EXTENSION IF NOT EXISTS postgres_fdw;
        DROP SERVER IF EXISTS src_srv CASCADE;
        CREATE SERVER src_srv
          FOREIGN DATA WRAPPER postgres_fdw
          OPTIONS (host 'pg_src', dbname 'demo', port '5432');
        CREATE USER MAPPING IF NOT EXISTS FOR dwh_user
          SERVER src_srv
          OPTIONS (user 'src_user', password 'src_pass');
        """,
    )

    # 3. Импорт исходной схемы bookings
    import_foreign = PostgresOperator(
        task_id="import_foreign_schema",
        postgres_conn_id="pg_dwh",
        sql="""
        DROP SCHEMA IF EXISTS src_fdw CASCADE;
        CREATE SCHEMA src_fdw;
        IMPORT FOREIGN SCHEMA bookings FROM SERVER src_srv INTO src_fdw;
        """,
    )

    # 4. Создание ODS
    create_ods = PostgresOperator(
        task_id="create_ods",
        postgres_conn_id="pg_dwh",
        sql="""
        CREATE TABLE IF NOT EXISTS ods.flights         (LIKE src_fdw.flights INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS ods.tickets         (LIKE src_fdw.tickets INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS ods.ticket_flights  (LIKE src_fdw.ticket_flights INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS ods.aircrafts       (LIKE src_fdw.aircrafts INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS ods.airports        (LIKE src_fdw.airports INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS ods.seats           (LIKE src_fdw.seats INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS ods.boarding_passes (LIKE src_fdw.boarding_passes INCLUDING ALL);
        """,
    )

    # 5. Загрузка ODS
    load_ods = PostgresOperator(
        task_id="load_ods",
        postgres_conn_id="pg_dwh",
        sql="""
        TRUNCATE ods.boarding_passes, ods.ticket_flights, ods.tickets, ods.seats,
                 ods.flights, ods.airports, ods.aircrafts;
        INSERT INTO ods.aircrafts       SELECT * FROM src_fdw.aircrafts;
        INSERT INTO ods.airports        SELECT * FROM src_fdw.airports;
        INSERT INTO ods.flights         SELECT * FROM src_fdw.flights;
        INSERT INTO ods.seats           SELECT * FROM src_fdw.seats;
        INSERT INTO ods.tickets         SELECT * FROM src_fdw.tickets;
        INSERT INTO ods.ticket_flights  SELECT * FROM src_fdw.ticket_flights;
        INSERT INTO ods.boarding_passes SELECT * FROM src_fdw.boarding_passes;
        """,
    )

    # 6. Справочники
    build_dims = PostgresOperator(
        task_id="build_dims",
        postgres_conn_id="pg_dwh",
        sql="""
        -- dim_date
        CREATE TABLE IF NOT EXISTS datamart.dim_date AS
        SELECT d::date AS calendar_date,
               EXTRACT(ISODOW FROM d)::INT AS day_of_week,
               EXTRACT(MONTH FROM d)::INT AS month,
               EXTRACT(QUARTER FROM d)::INT AS quarter,
               EXTRACT(YEAR FROM d)::INT AS year,
               FALSE AS is_holiday
        FROM generate_series(
            (SELECT MIN(date(scheduled_departure)) FROM ods.flights),
            (SELECT MAX(date(scheduled_departure)) FROM ods.flights),
            interval '1 day'
        ) d
        ON CONFLICT DO NOTHING;

        -- dim_time
        CREATE TABLE IF NOT EXISTS datamart.dim_time AS
        SELECT h.hh, m.mi,
               CASE
                 WHEN h.hh BETWEEN 5  AND 11 THEN 'morning'
                 WHEN h.hh BETWEEN 12 AND 16 THEN 'day'
                 WHEN h.hh BETWEEN 17 AND 21 THEN 'evening'
                 ELSE 'night'
               END AS part_of_day
        FROM generate_series(0,23) h(hh)
        CROSS JOIN generate_series(0,59) m(mi);

        -- dim_airport
        CREATE TABLE IF NOT EXISTS datamart.dim_airport AS
        SELECT DISTINCT airport_code, airport_name, city, timezone, longitude, latitude
        FROM ods.airports;

        -- dim_aircraft
        CREATE TABLE IF NOT EXISTS datamart.dim_aircraft AS
        SELECT DISTINCT aircraft_code, model, range
        FROM ods.aircrafts;

        -- dim_airline
        CREATE TABLE IF NOT EXISTS datamart.dim_airline AS
        SELECT DISTINCT LEFT(TRIM(flight_no),2) AS airline_code
        FROM ods.flights
        WHERE flight_no IS NOT NULL;

        -- dim_route
        CREATE TABLE IF NOT EXISTS datamart.dim_route AS
        SELECT DISTINCT f.departure_airport AS origin_airport,
                        f.arrival_airport   AS destination_airport
        FROM ods.flights f
        WHERE f.departure_airport <> f.arrival_airport;
        """,
    )

    # 7. Факт
    build_fact = PostgresOperator(
        task_id="build_fact",
        postgres_conn_id="pg_dwh",
        sql="""
        DROP TABLE IF EXISTS datamart.fact_flights CASCADE;
        CREATE TABLE datamart.fact_flights AS
        SELECT f.flight_id,
               date(f.scheduled_departure) AS date_id,
               f.departure_airport,
               LEFT(TRIM(f.flight_no),2)   AS airline_code,
               f.arrival_airport,
               COUNT(DISTINCT bp.ticket_no)        AS passenger_count,
               COUNT(DISTINCT tf.ticket_no)        AS tickets_sold,
               COALESCE(SUM(tf.amount),0)::NUMERIC(12,2) AS revenue
        FROM ods.flights f
        LEFT JOIN ods.ticket_flights tf ON tf.flight_id = f.flight_id
        LEFT JOIN ods.boarding_passes bp ON bp.flight_id = f.flight_id
        GROUP BY f.flight_id, f.scheduled_departure, f.departure_airport, f.arrival_airport, f.flight_no
        ORDER BY f.flight_id;
        """,
    )

    create_schemas >> fdw_setup >> import_foreign >> create_ods >> load_ods >> build_dims >> build_fact
