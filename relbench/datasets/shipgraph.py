import os
import zipfile

import numpy as np
import pandas as pd
import pooch
from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor


class ShipGraphDataSet(Dataset):
    train_timestamp = pd.Timestamp("2025-10-16 18:00:00")
    train_timestamp_end = pd.Timestamp("2025-10-16 18:30:00")

    val_timestamp = pd.Timestamp("2025-10-17 18:00:00")
    val_timestamp_end = pd.Timestamp("2025-10-17 18:30:00")

    test_timestamp = pd.Timestamp("2025-10-19 09:00:00")
    test_timestamp_end = pd.Timestamp("2025-10-19 09:30:00")

    def _ensure_datetime(self, df, columns):
        """Helper function to convert columns to datetime only if needed."""
        for col in columns:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        return df

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""

        # S3 path
        s3_path = 's3://relgt-data-export/processed/'
        
        # Read parquet files from S3
        deliveries = pd.read_parquet(os.path.join(s3_path, "deliveries.parquet"))
        delivery_edges = pd.read_parquet(os.path.join(s3_path, "delivery_edges.parquet"))
        exit202_edges = pd.read_parquet(os.path.join(s3_path, "exit202_edges.parquet"))
        induct_edges = pd.read_parquet(os.path.join(s3_path, "induct_edges.parquet"))
        missort_edges = pd.read_parquet(os.path.join(s3_path, "missort_edges.parquet"))
        packages = pd.read_parquet(os.path.join(s3_path, "packages.parquet"))
        sort_centers = pd.read_parquet(os.path.join(s3_path, "sort_centers.parquet"))
        linehaul_edges = pd.read_parquet(os.path.join(s3_path, "linehaul_edges.parquet"))
        problem_edges = pd.read_parquet(os.path.join(s3_path, "problem_edges.parquet"))
        
        # Convert timestamps conditionally (only if not already datetime)
        packages = self._ensure_datetime(packages, ['creation_date', 'min_event_time', 'max_event_time', 'pdd', 'delivered_date'])
        delivery_edges = self._ensure_datetime(delivery_edges, ['event_time', 'plan_time'])
        exit202_edges = self._ensure_datetime(exit202_edges, ['event_time', 'plan_time'])
        induct_edges = self._ensure_datetime(induct_edges, ['event_time', 'plan_time'])
        missort_edges = self._ensure_datetime(missort_edges, ['event_time'])
        linehaul_edges = self._ensure_datetime(linehaul_edges, ['event_time', 'plan_time'])
        #problem_edges = self._ensure_datetime(problem_edges, ['event_time'])

        # Remove columns that are irrelevant, leak time,
        # or have too many missing values

        # Drop the Wikipedia URL and some time columns with many missing values
        exit202_edges.drop(
            columns=["parent_container_id"],
            inplace=True,
            errors='ignore'
        )

        # Drop the Wikipedia URL as it is unique for each row
        packages.drop(
            columns=["id","vertex_id", "has_cycle", "current_node", "is_delivered", "container_id","carrier_container_id", "current_node","is_return","vehicle_run_id"],
            inplace=True,
            errors='ignore'
        )

        # Drop from_vertex_id and to_vertex_id from all edge tables
        delivery_edges.drop(columns=['to_vertex'], inplace=True, errors='ignore')
        exit202_edges.drop(columns=['to_vertex'], inplace=True, errors='ignore')
        induct_edges.drop(columns=['to_vertex'], inplace=True, errors='ignore')
        missort_edges.drop(columns=['to_vertex'], inplace=True, errors='ignore')
        problem_edges.drop(columns=['to_vertex'], inplace=True, errors='ignore')
        linehaul_edges.drop(columns=['to_vertex_id'], inplace=True, errors='ignore')

        # Deduplicate on pk column for all dataframes
        packages = packages.drop_duplicates(subset=['pk'], keep='first')
        deliveries = deliveries.drop_duplicates(subset=['pk'], keep='first')
        delivery_edges = delivery_edges.drop_duplicates(subset=['pk'], keep='first')
        exit202_edges = exit202_edges.drop_duplicates(subset=['pk'], keep='first')
        induct_edges = induct_edges.drop_duplicates(subset=['pk'], keep='first')
        missort_edges = missort_edges.drop_duplicates(subset=['pk'], keep='first')
        problem_edges = problem_edges.drop_duplicates(subset=['pk'], keep='first')
        linehaul_edges = linehaul_edges.drop_duplicates(subset=['pk'], keep='first')
        sort_centers = sort_centers.drop_duplicates(subset=['pk'], keep='first')
     
        tables = {}

        tables["packages"] = Table(
            df=pd.DataFrame(packages),
            fkey_col_to_pkey_table={},
            pkey_col="pk",
            time_col="creation_date",
        )

        tables["deliveries"] = Table(
            df=pd.DataFrame(deliveries),
            fkey_col_to_pkey_table={},
            pkey_col="pk",
            time_col=None,
        )

        tables["delivery_edges"] = Table(
            df=pd.DataFrame(delivery_edges),
            fkey_col_to_pkey_table={
                "from_pk":"packages",
                "to_pk": "deliveries" 
                },
            pkey_col='pk',
            time_col="event_time",
        )

        tables["exit202_edges"] = Table(
            df=pd.DataFrame(exit202_edges),
            fkey_col_to_pkey_table={
                "from_pk": "packages",
                "to_pk": "sort_centers",
            },
            pkey_col="pk",
            time_col="event_time",
        )

        tables["induct_edges"] = Table(
            df=pd.DataFrame(induct_edges),
            fkey_col_to_pkey_table={
                "from_pk": "packages",
                "to_pk": "sort_centers",
            },
            pkey_col="pk",
            time_col="event_time",
        )

        tables["missort_edges"] = Table(
            df=pd.DataFrame(missort_edges),
            fkey_col_to_pkey_table={
                "from_pk": "packages",
                "to_pk": "sort_centers",
            },
            pkey_col="pk",
            time_col="event_time",
        )

        tables["problem_edges"] = Table(
            df=pd.DataFrame(problem_edges),
            fkey_col_to_pkey_table={
                "from_pk": "packages",
                "to_pk": "sort_centers",
            },
            pkey_col="pk",
            time_col="event_time",
        )

        tables["linehaul_edges"] = Table(
            df=pd.DataFrame(linehaul_edges),
            fkey_col_to_pkey_table={
                "from_pk": "packages",
                "to_pk": "sort_centers",
            },
            pkey_col="pk",
            time_col="event_time",
        )

        tables["sort_centers"] = Table(
            df=pd.DataFrame(sort_centers),
            fkey_col_to_pkey_table={},
            pkey_col="pk",
            time_col=None,
        )

        return Database(tables)