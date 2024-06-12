import datetime
import time
from dotenv import load_dotenv
import os
import joblib
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently import ColumnMapping
from prefect import flow, task

import psycopg

load_dotenv()

NUMERICAL = [
    'passenger_count',
    'trip_distance',
    'fare_amount',
    'total_amount'
]

CATEGORICAL = [
    'PULocationID',
    'DOLocationID'
]

COL_MAPPING = ColumnMapping(
    prediction='prediction',
    numerical_features=NUMERICAL,
    categorical_features=CATEGORICAL,
    target=None
)

CONNECT_STRING = f'host={os.getenv("POSTGRES_HOST")} port={os.getenv("POSTGRES_PORT")} user={os.getenv("POSTGRES_USER")} password={os.getenv("POSTGRES_PASSWORD")}'

@task(name='Prepare Database')
def prep_db():
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics(
        timestamp timestamp,
        prediction_drift float,
        num_drifted_columns integer,
        share_missing_values float
    );
    """

    with psycopg.connect(CONNECT_STRING, autocommit=True) as conn:
        # zoek naar database genaamd 'test' in de metadata van postgres
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("CREATE DATABASE test;")
        with psycopg.connect(f'{CONNECT_STRING} dbname=test') as conn:
            conn.execute(create_table_query)

@task(name='Load Data & Model')
def prep_data():
    ref_data = pd.read_parquet('data/reference.parquet')
    with open('models/lin_reg.bin', 'rb') as f_in:
        model = joblib.load(f_in)
    
    raw_data = pd.read_parquet('data/green_tripdata_2022-02.parquet')

    return ref_data, model, raw_data

@flow(name='Calculate Metrics')
def calculate_metrics(current_data, model, ref_data):
    current_data['prediction'] = model.predict(current_data[NUMERICAL + CATEGORICAL].fillna(0))

    report = create_report(current_data, ref_data)

    prediction_drift = report['metrics'][0]['result']['drift_score']
    num_drifted_cols = report['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_vals = report['metrics'][2]['result']['current']['share_of_missing_values']

    return prediction_drift, num_drifted_cols, share_missing_vals

@task(name='Create Report')
def create_report(current_data, ref_data):
    report = Report(metrics = [
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(
        reference_data=ref_data,
        current_data=current_data,
        column_mapping=COL_MAPPING
    )

    result = report.as_dict()
    return result

@task(name='Save Metrics To DB')
def save_metrics_to_db(cursor, date, prediction_drift, num_drifted_cols, share_missing_vals):
    cursor.execute("""
    INSERT INTO metrics(
        timestamp,
        prediction_drift,
        num_drifted_columns,
        share_missing_values
    )
    VALUES (%s, %s, %s, %s);
    """, (date, prediction_drift, num_drifted_cols, share_missing_vals))
    
@flow(name='Monitor Flow')
def monitor():
    startDate = datetime.datetime(2022, 2, 1, 0, 0)
    endDate = datetime.datetime(2022, 2, 2, 0, 0)

    prep_db()

    ref_data, model, raw_data = prep_data()


    with psycopg.connect(f'{CONNECT_STRING} dbname=test') as conn:
        with conn.cursor() as cursor:
            # get daily data to simulate rides in february
            for i in range(0, 27):
                current_data = raw_data[(raw_data.lpep_pickup_datetime >= startDate) &
                                        (raw_data.lpep_pickup_datetime < endDate)]
                
                create_metrics(startDate, ref_data, model, cursor, current_data)

                startDate += datetime.timedelta(1)
                endDate += datetime.timedelta(1)

                time.sleep(1)
                print("data added")

@flow()
def create_metrics(startDate, ref_data, model, cursor, current_data):
    prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(current_data, model, ref_data)
    save_metrics_to_db(cursor, startDate, prediction_drift, num_drifted_cols, share_missing_vals)

if __name__ == '__main__':
    monitor()