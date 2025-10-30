import os
import zipfile

import numpy as np
import pandas as pd
import pooch
from relbench.base import Database, Dataset, Table
from relbench.utils import unzip_processor


class ShipGraphDataSet(Dataset):
    val_timestamp = pd.Timestamp("2025-10-17 18:00:00")
    test_timestamp = pd.Timestamp("2025-10-18 09:00:00")

    def make_db(self) -> Database:
        r"""Process the raw files into a database."""

        path = '/home/ubuntu/.cache/shipgraph/'
        #zip_path = os.path.join(path, "db.zip")
        
        # Unzip the file if it exists
        #if os.path.exists(zip_path):
        #    print(f"Unzipping {zip_path}...")
        #    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #        zip_ref.extractall(path)
        #    print(f"Extracted to {path}")
        
        deliveries = pd.read_csv(os.path.join(path, "deliveries.csv"))
        delivery_edges = pd.read_csv(os.path.join(path, "delivery_edges.csv"))
        exit202_edges = pd.read_csv(os.path.join(path, "exit202_edges.csv"))
        induct_edges = pd.read_csv(os.path.join(path, "induct_edges.csv"))
        missort_edges = pd.read_csv(os.path.join(path, "missort_edges.csv"))
        packages = pd.read_csv(os.path.join(path, "packages.csv"))
        sort_centers = pd.read_csv(os.path.join(path, "sort_centers.csv"))
        linehaul_edges = pd.read_csv(os.path.join(path, "linehaul_edges.csv"))
        problem_edges = pd.read_csv(os.path.join(path, "problem_edges.csv"))
        
        # Convert timestamps - creation_date is in ms, but min/max_event_time are ISO strings
        packages['creation_date'] = pd.to_datetime(packages['creation_date'], unit='ms')
        packages['min_event_time'] = pd.to_datetime(packages['min_event_time'])
        packages['max_event_time'] = pd.to_datetime(packages['max_event_time'])

        # Replace commas with -> in plan_route if it exists
        if 'plan_route' in packages.columns:
            packages['plan_route'] = packages['plan_route'].str.replace(',', '->', regex=False)

        # Convert event_time for all edge dataframes
        delivery_edges['event_time'] = pd.to_datetime(delivery_edges['event_time'])
        exit202_edges['event_time'] = pd.to_datetime(exit202_edges['event_time'])
        induct_edges['event_time'] = pd.to_datetime(induct_edges['event_time'])
        missort_edges['event_time'] = pd.to_datetime(missort_edges['event_time'])
        linehaul_edges['event_time'] = pd.to_datetime(linehaul_edges['event_time'])
        problem_edges['event_time'] = pd.to_datetime(problem_edges['event_time'])

        delivery_edges['plan_time'] = pd.to_datetime(delivery_edges['plan_time'])
        exit202_edges['plan_time'] = pd.to_datetime(exit202_edges['plan_time'])
        induct_edges['plan_time'] = pd.to_datetime(induct_edges['plan_time'])
        linehaul_edges['plan_time'] = pd.to_datetime(linehaul_edges['plan_time'])
        

        # Remove columns that are irrelevant, leak time,
        # or have too many missing values

        # Drop the Wikipedia URL and some time columns with many missing values
        exit202_edges.drop(
            columns=["parent_container_id"],
            inplace=True,
        )

        # Drop the Wikipedia URL as it is unique for each row
        packages.drop(
            columns=["id", "has_cycle", "current_node", "is_delivered", "container_id"],
            inplace=True,
        )

        # Drop from_vertex_id and to_vertex_id from all edge tables
        delivery_edges.drop(columns=['from_vertex_id', 'to_vertex_id'], inplace=True, errors='ignore')
        exit202_edges.drop(columns=['from_vertex_id', 'to_vertex_id'], inplace=True, errors='ignore')
        induct_edges.drop(columns=['from_vertex_id', 'to_vertex_id'], inplace=True, errors='ignore')
        missort_edges.drop(columns=['from_vertex_id', 'to_vertex_id'], inplace=True, errors='ignore')
        problem_edges.drop(columns=['from_vertex_id', 'to_vertex_id'], inplace=True, errors='ignore')
        linehaul_edges.drop(columns=['from_vertex_id', 'to_vertex_id'], inplace=True, errors='ignore')

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