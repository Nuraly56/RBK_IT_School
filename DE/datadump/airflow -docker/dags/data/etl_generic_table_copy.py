from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime
import csv, os

# === НАСТРОЙ ПОД СЕБЯ ===
SRC_CONN_ID = "src_db"              # твой коннект к источнику
DWH_CONN_ID = "postgres_mb_dwh"     # коннект к DWH
DATA_DIR = "/opt/airflow/dags/data"
CSV_PATH = os.path.join(DATA_DIR, "src_table.csv")

EXTRACT_SQL = """
SELECT
  o.id           AS order_id,
  o.user_id      AS user_id,
  o.created_at   AS created_at,
  o.total_amount AS total_amount,
  o.status       AS status
FROM public.orders o
"""  # <<< ЗАМЕНИ на свою таблицу/поля

COLUMNS = [                       # имена и типы в DWH (в том же порядке!)
  ("order_id","BIGINT"),
  ("user_id","BIGINT"),
  ("created_at","TIMESTAMP"),
  ("total_amount","NUMERIC(12,2)"),
  ("status","TEXT"),
]
KEY_COLS = ["order_id"]           # ключ для upsert

STG_TABLE = "dwh.stg_orders"
FACT_TABLE = "dwh.fact_orders"

# === ДАЛЬШЕ НЕ ТРОГАЕМ ===
DDL_DWH = f"""
CREATE SCHEMA IF NOT EXISTS dwh;
CREATE UNLOGGED TABLE IF NOT EXISTS {STG_TABLE} (
  {', '.join([f'{n} {t}' for n,t in COLUMNS])}
);
CREATE TABLE IF NOT EXISTS {FACT_TABLE} (
  {', '.join([f'{n} {t}' for n,t in COLUMNS])},
  PRIMARY KEY ({', '.join(KEY_COLS)})
);
"""

def extract_to_csv(**_):
    os.makedirs(DATA_DIR, exist_ok=True)
    src = PostgresHook(postgres_conn_id=SRC_CONN_ID)
    conn = src.get_conn()
    cur = conn.cursor(name="stream_cur")
    cur.itersize = 50000
    cur.execute(EXTRACT_SQL)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow([c[0] for c in COLUMNS])
        for row in cur: w.writerow(row)
    cur.close(); conn.close()

def copy_csv_to_staging(**_):
    dwh = PostgresHook(postgres_conn_id=DWH_CONN_ID)
    dwh.run(f"TRUNCATE TABLE {STG_TABLE};")
    copy_sql = f"COPY {STG_TABLE} ({', '.join([c[0] for c in COLUMNS])}) FROM STDIN WITH CSV HEADER"
    dwh.copy_expert(sql=copy_sql, filename=CSV_PATH)

UPSERT_SQL = f"""
INSERT INTO {FACT_TABLE} ({', '.join([c[0] for c in COLUMNS])})
SELECT {', '.join([c[0] for c in COLUMNS])} FROM {STG_TABLE}
ON CONFLICT ({', '.join(KEY_COLS)}) DO UPDATE
SET {', '.join([f"{c[0]} = EXCLUDED.{c[0]}" for c in COLUMNS if c[0] not in KEY_COLS])};
"""

default_args = {"owner":"dataeng","retries":0}
with DAG(
    dag_id="etl_generic_table_copy",
    default_args=default_args,
    start_date=datetime(2025,8,1),
    schedule_interval=None,
    catchup=False,
    tags=["etl","copy","dwh"],
) as dag:
    create = PostgresOperator(task_id="create_schema_and_tables", postgres_conn_id=DWH_CONN_ID, sql=DDL_DWH)
    extract = PythonOperator(task_id="extract_to_csv", python_callable=extract_to_csv)
    load_stg = PythonOperator(task_id="copy_csv_to_staging", python_callable=copy_csv_to_staging)
    upsert = PostgresOperator(task_id="upsert_to_fact", postgres_conn_id=DWH_CONN_ID, sql=UPSERT_SQL)
    create >> extract >> load_stg >> upsert
