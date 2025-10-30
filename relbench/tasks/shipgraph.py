import duckdb
import pandas as pd

from relbench.base import Database, EntityTask, Table, TaskType
from relbench.metrics import accuracy, average_precision, f1, mae, r2, rmse, roc_auc


class DeliveryTimePrediction(EntityTask):
    task_type = TaskType.REGRESSION
    entity_col = "from_pk"
    entity_table = "packages"
    target_col = "gap"
    timedelta = pd.Timedelta(hours=2)
    metrics = [r2, mae, rmse]

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]", delivery_time_start, delivery_time_end) -> Table:
        timestamp_df = pd.DataFrame({"timestamp": timestamps})
        timestamp_df = timestamp_df.sort_values('timestamp')
        
        packages = db.table_dict["packages"].df
        delivery_edges = db.table_dict["delivery_edges"].df
        exit202_edges = db.table_dict["exit202_edges"].df
        induct_edges = db.table_dict["induct_edges"].df
        #missort_edges = db.table_dict["missort_edges"].df
        #problem_edges = db.table_dict["problem_edges"].df
        linehaul_edges = db.table_dict["linehaul_edges"].df

        df = duckdb.sql(
        f"""
            WITH all_edges AS (
                SELECT 
                    from_pk,
                    event_time,
                    plan_time,
                    gap,
                    'delivery' AS edge_name
                FROM delivery_edges
                
                UNION ALL
                
                SELECT 
                    from_pk,
                    event_time,
                    plan_time,
                    gap,
                    'exit202' AS edge_name
                FROM exit202_edges
                
                UNION ALL
                
                SELECT 
                    from_pk,
                    event_time,
                    plan_time,
                    gap,
                    'induct' AS edge_name
                FROM induct_edges
                
                UNION ALL
                            
                SELECT 
                    from_pk,
                    event_time,
                    plan_time,
                    gap,
                    'linehaul' AS edge_name
                FROM linehaul_edges
            ),
            timestamp_edge_pairs AS (
                SELECT 
                    t.timestamp,
                    e.from_pk,
                    e.event_time,
                    e.gap,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.timestamp, e.from_pk 
                        ORDER BY e.event_time ASC
                    ) AS rn
                FROM timestamp_df t
                INNER JOIN all_edges e 
                    ON e.event_time >= t.timestamp 
                INNER JOIN packages p ON p.pk = e.from_pk
                WHERE p.delivered_date >= '{delivery_time_start}'
                    AND p.delivered_date <= '{delivery_time_end}'
                    AND p.min_event_time >= t.timestamp
                    AND p.max_event_time >= t.timestamp 
            )
            SELECT
                timestamp,
                from_pk,
                gap
            FROM timestamp_edge_pairs
            WHERE rn = 1 
            ORDER BY timestamp ASC
            """
        ).df()
        
        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.entity_col: self.entity_table,
            },
            pkey_col=None,
            time_col="timestamp",
        )