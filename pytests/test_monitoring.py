import datetime
from io import BytesIO
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest

from evidently_flow import (
    CATEGORICAL,
    CONNECT_STRING,
    NUMERICAL,
    calculate_metrics,
    monitor,
    prep_data,
    prep_db,
    save_metrics_to_db,
)

# Mock data for testing
mock_ref_data = pd.DataFrame(
    {
        "passenger_count": [1, 2, 1],
        "trip_distance": [1.0, 2.0, 1.5],
        "fare_amount": [10.0, 20.0, 15.0],
        "total_amount": [12.0, 24.0, 18.0],
        "PULocationID": [1, 2, 3],
        "DOLocationID": [4, 5, 6],
    }
)

mock_current_data = pd.DataFrame(
    {
        "lpep_pickup_datetime": pd.to_datetime(
            ["2022-02-01", "2022-02-01", "2022-02-01"]
        ),
        "passenger_count": [1, 2, 1],
        "trip_distance": [1.0, 2.0, 1.5],
        "fare_amount": [10.0, 20.0, 15.0],
        "total_amount": [12.0, 24.0, 18.0],
        "PULocationID": [1, 2, 3],
        "DOLocationID": [4, 5, 6],
    }
)

mock_model = MagicMock()
mock_model.predict.return_value = [15.0, 30.0, 22.5]

# Mock report result
mock_report_result = {
    "metrics": [
        {"result": {"drift_score": 0.1}},
        {"result": {"number_of_drifted_columns": 1}},
        {"result": {"current": {"share_of_missing_values": 0.01}}},
    ]
}


@patch("evidently_flow.psycopg.connect")
def test_prep_db(mock_connect):
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    prep_db()

    assert mock_cursor.execute.call_count == 0


@patch("evidently_flow.pd.read_parquet")
@patch("evidently_flow.joblib.load")
def test_prep_data(mock_joblib_load, mock_read_parquet):
    mock_read_parquet.side_effect = [mock_ref_data, mock_current_data]
    mock_joblib_load.return_value = mock_model

    ref_data, model, raw_data = prep_data()

    assert ref_data.equals(mock_ref_data)
    assert raw_data.equals(mock_current_data)
    assert model == mock_model


@patch("evidently_flow.create_report")
def test_calculate_metrics(mock_create_report):
    mock_create_report.return_value = mock_report_result

    prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(
        mock_current_data, mock_model, mock_ref_data
    )

    assert prediction_drift == 0.1
    assert num_drifted_cols == 1
    assert share_missing_vals == 0.01

@patch("evidently_flow.psycopg.connect")
def test_save_metrics_to_db(mock_connect):
    mock_conn = mock_connect.return_value.__enter__.return_value
    mock_cursor = mock_conn.cursor.return_value.__enter__.return_value

    cursor = mock_cursor
    date = datetime.datetime(2022, 2, 1, 0, 0)
    prediction_drift = 0.1
    num_drifted_cols = 1
    share_missing_vals = 0.01

    save_metrics_to_db(
        cursor, date, prediction_drift, num_drifted_cols, share_missing_vals
    )

    mock_cursor.execute.assert_called_once()


@patch("evidently_flow.psycopg.connect")
@patch("evidently_flow.prep_data")
@patch("evidently_flow.prep_db")
@patch("evidently_flow.create_metrics")
def test_monitor(mock_create_metrics, mock_prep_db, mock_prep_data, mock_connect):
    mock_prep_data.return_value = (mock_ref_data, mock_model, mock_current_data)

    monitor()

    assert mock_prep_db.called
    assert mock_prep_data.called
    assert mock_create_metrics.call_count == 27
