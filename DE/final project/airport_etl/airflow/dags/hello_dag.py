from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator

with DAG(
    dag_id="hello_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["test"],
):
    start = EmptyOperator(task_id="start")
    done = EmptyOperator(task_id="done")
    start >> done
