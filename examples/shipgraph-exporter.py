import pandas as pd
from gremlin_python.driver import client, serializer
from datetime import datetime, timedelta
import boto3
from io import StringIO, BytesIO
import zipfile
import nest_asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import threading
from collections import defaultdict
import json

# Apply nest_asyncio to fix event loop issues
nest_asyncio.apply()

# ===================== CONFIGURATION =====================
NEPTUNE_ENDPOINT = "swa-shipgraph-neptune-instance-prod-us-east-1.c6fskces27nt.us-east-1.neptune.amazonaws.com"
NEPTUNE_PORT = 8182

# Date ranges for training/validation/test splits (inclusive on both bounds)
TRAINING_START_DATE = "2025-10-16T18:00:00.000Z"
TRAINING_END_DATE = "2025-10-16T18:30:00.999Z"

VALIDATION_START_DATE = "2025-10-17T18:00:00.000Z"
VALIDATION_END_DATE = "2025-10-17T18:30:00.000Z"

TEST_START_DATE = "2025-10-19T09:00:00.000Z"
TEST_END_DATE = "2025-10-19T09:30:00.000Z"

# Worker settings
PACKAGE_FETCH_BATCH_SIZE = 5000  # Batch size for fetching packages
PACKAGE_BATCH_SIZE = 5000        # Batch size for fetching edges per package group
MAX_WORKERS = 30                 # Number of parallel queries

# Local Configuration
LOCAL_DIR = "/home/ubuntu/.cache/shipgraph"

# ===================== THREAD-SAFE COUNTER =====================
class ThreadSafeCounter:
    """Thread-safe counter for tracking progress"""
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

# ===================== CONNECTION =====================
def get_neptune_client():
    """Create Neptune Gremlin client"""
    neptune_endpoint = f'wss://{NEPTUNE_ENDPOINT}:{NEPTUNE_PORT}/gremlin'
    return client.Client(
        neptune_endpoint,
        'g',
        message_serializer=serializer.GraphSONSerializersV2d0()
    )

# ===================== DATA PROCESSING =====================

def clean_plan_route(value):
    """Convert JSON array string to -> separated string"""
    if not value or value == '':
        return ''
    
    try:
        # If it's a JSON array string, parse and join with ->
        if isinstance(value, str) and value.startswith('['):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return '->'.join([str(item) for item in parsed])
        return value
    except:
        return value

def clean_container_problems(value):
    """Convert JSON array string to space-separated string"""
    if not value or value == '':
        return ''
    
    try:
        # If it's a JSON array string, parse and join with spaces
        if isinstance(value, str) and value.startswith('['):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return ' '.join([str(item) for item in parsed])
        return value
    except:
        return value

def flatten_value(value):
    """Flatten Neptune property values (which are returned as lists)"""
    if isinstance(value, list):
        if len(value) == 0:
            return ''
        elif len(value) == 1:
            val = value[0]
            # Handle datetime objects
            if isinstance(val, datetime):
                return val.isoformat()
            return val
        else:
            # Multi-valued property - join with semicolon
            processed = []
            for v in value:
                if isinstance(v, datetime):
                    processed.append(v.isoformat())
                else:
                    processed.append(str(v))
            return ';'.join(processed)
    elif isinstance(value, datetime):
        return value.isoformat()
    else:
        return value

# ===================== OPTIMIZED PARALLEL PACKAGE FETCH FUNCTIONS =====================

def estimate_package_count(start_date, end_date, dataset_name):
    """Estimate total package count for a date range (excluding returns)"""
    print(f"[{dataset_name}] Estimating package count (excluding returns)...")
    gremlin_client = get_neptune_client()
    start_time = time.time()
    
    try:
        query = f"""
        g.V().hasLabel('Package')
         .has('delivered_date', gte(datetime('{start_date}')))
         .has('delivered_date', lte(datetime('{end_date}')))
         .not(has('is_return', true))
         .count()
        """
        
        callback = gremlin_client.submit(query)
        count = callback.all().result()[0]
        elapsed = time.time() - start_time
        print(f"[{dataset_name}] ✓ Estimated {count:,} packages (no returns) in {elapsed:.2f}s")
        return count
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{dataset_name}] ✗ Count error after {elapsed:.2f}s: {str(e)[:200]}")
        return 0
    finally:
        gremlin_client.close()

def fetch_packages_batch_tagged(start_date, end_date, skip_count, limit, dataset_name, batch_id):
    """Fetch a batch of packages with pagination and return tagged results (excluding returns)"""
    print(f"[{dataset_name}] Batch #{batch_id}: Fetching skip={skip_count}, limit={limit}...")
    gremlin_client = get_neptune_client()
    start_time = time.time()
    
    try:
        query = f"""
        g.V().hasLabel('Package')
         .has('delivered_date', gte(datetime('{start_date}')))
         .has('delivered_date', lte(datetime('{end_date}')))
         .not(has('is_return', true))
         .range({skip_count}, {skip_count + limit})
         .project('vertex_id', 'properties')
         .by(id())
         .by(valueMap())
        """
        
        callback = gremlin_client.submit(query)
        results = callback.all().result()
        
        # Flatten the structure
        flattened_results = []
        for pkg in results:
            flat_pkg = {'vertex_id': pkg['vertex_id'], 'dataset': dataset_name}
            if pkg.get('properties'):
                for key, value in pkg['properties'].items():
                    flattened = flatten_value(value)
                    # Special handling for plan_route
                    if key == 'plan_route':
                        flat_pkg[key] = clean_plan_route(flattened)
                    else:
                        flat_pkg[key] = flattened
            flattened_results.append(flat_pkg)
        
        elapsed = time.time() - start_time
        print(f"[{dataset_name}] Batch #{batch_id}: ✓ {len(flattened_results):,} packages in {elapsed:.2f}s")
        return {
            'dataset': dataset_name,
            'batch_id': batch_id,
            'packages': flattened_results,
            'count': len(flattened_results)
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{dataset_name}] Batch #{batch_id}: ✗ Error after {elapsed:.2f}s: {str(e)[:200]}")
        return {
            'dataset': dataset_name,
            'batch_id': batch_id,
            'packages': [],
            'count': 0,
            'error': str(e)
        }
    finally:
        gremlin_client.close()

def fetch_all_packages_parallel(datasets_config, batch_size=5000, max_workers=30):
    """Fetch ALL package batches across ALL datasets in parallel (THREAD-SAFE)"""
    print("\n" + "="*70)
    print("PARALLEL PACKAGE FETCHING - ALL DATASETS (THREAD-SAFE)")
    print("="*70)
    print(f"Datasets: {len(datasets_config)}")
    print(f"Batch size: {batch_size:,}")
    print(f"Max workers: {max_workers}")
    print("Filtering: Excluding packages with is_return=true")
    print("="*70)
    
    # Step 1: Estimate counts for all datasets
    print("\n[Step 1] Estimating package counts...")
    with ThreadPoolExecutor(max_workers=len(datasets_config)) as executor:
        count_futures = {
            executor.submit(
                estimate_package_count,
                config['start_date'],
                config['end_date'],
                config['name']
            ): config['name']
            for config in datasets_config
        }
        
        dataset_counts = {}
        for future in as_completed(count_futures):
            dataset_name = count_futures[future]
            try:
                count = future.result()
                dataset_counts[dataset_name] = count
            except Exception as e:
                print(f"[{dataset_name}] ✗ Count failed: {e}")
                dataset_counts[dataset_name] = 0
    
    # Step 2: Create all batch tasks
    print("\n[Step 2] Creating batch tasks...")
    all_tasks = []
    
    for config in datasets_config:
        dataset_name = config['name']
        estimated_count = dataset_counts.get(dataset_name, 0)
        
        if estimated_count == 0:
            print(f"[{dataset_name}] ⚠️  Estimated 0 packages, will still try 1 batch")
            all_tasks.append({
                'start_date': config['start_date'],
                'end_date': config['end_date'],
                'skip_count': 0,
                'limit': batch_size,
                'dataset_name': dataset_name,
                'batch_id': 1
            })
            continue
        
        num_batches = (estimated_count + batch_size - 1) // batch_size
        print(f"[{dataset_name}] Planning {num_batches} batches for ~{estimated_count:,} packages")
        
        for batch_id in range(num_batches):
            skip_count = batch_id * batch_size
            all_tasks.append({
                'start_date': config['start_date'],
                'end_date': config['end_date'],
                'skip_count': skip_count,
                'limit': batch_size,
                'dataset_name': dataset_name,
                'batch_id': batch_id + 1
            })
    
    total_tasks = len(all_tasks)
    print(f"\n[Step 3] Fetching {total_tasks} batches in parallel...")
    print(f"Using {max_workers} parallel workers")
    print("="*70)
    
    all_results = []
    results_lock = threading.Lock()
    completed_counter = ThreadSafeCounter()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                fetch_packages_batch_tagged,
                task['start_date'],
                task['end_date'],
                task['skip_count'],
                task['limit'],
                task['dataset_name'],
                task['batch_id']
            ): task
            for task in all_tasks
        }
        
        for future in as_completed(futures):
            task = futures[future]
            
            try:
                result = future.result()
                
                with results_lock:
                    all_results.append(result)
                
                completed = completed_counter.increment()
                
                if completed % 10 == 0 or completed == total_tasks:
                    progress_pct = completed * 100 // total_tasks
                    print(f"\n>>> Progress: {completed}/{total_tasks} batches completed ({progress_pct}%)")
                    
            except Exception as e:
                print(f"[{task['dataset_name']}] Batch #{task['batch_id']} failed: {e}")
    
    # Step 4: Organize results by dataset
    print("\n" + "="*70)
    print("ORGANIZING RESULTS")
    print("="*70)
    
    organized_packages = defaultdict(list)
    
    for result in all_results:
        dataset_name = result['dataset']
        packages = result['packages']
        organized_packages[dataset_name].extend(packages)
    
    organized_packages = dict(organized_packages)
    
    # Print summary
    print("\n" + "="*70)
    print("FETCH SUMMARY")
    print("="*70)
    total_packages = 0
    for dataset_name in ['training', 'validation', 'test']:
        packages = organized_packages.get(dataset_name, [])
        count = len(packages)
        total_packages += count
        print(f"  {dataset_name:12s}: {count:,} packages")
    print(f"  {'TOTAL':12s}: {total_packages:,} packages")
    print("="*70)
    
    return organized_packages

# ===================== NODE FETCH FUNCTIONS =====================

def fetch_all_sort_centers():
    """Fetch ALL sort centers (no time filter)"""
    print(f"[SortCenters] Starting fetch (no time filter)...")
    gremlin_client = get_neptune_client()
    start_time = time.time()
    
    try:
        query = """
        g.V().hasLabel('SortCenter')
         .project('vertex_id', 'properties')
         .by(id())
         .by(valueMap())
        """
        
        callback = gremlin_client.submit(query)
        results = callback.all().result()
        
        # Flatten
        flattened = []
        for node in results:
            flat_node = {'vertex_id': node['vertex_id']}
            if node.get('properties'):
                for key, value in node['properties'].items():
                    flat_node[key] = flatten_value(value)
            flattened.append(flat_node)
        
        elapsed = time.time() - start_time
        print(f"[SortCenters] ✓ Fetched {len(flattened):,} records in {elapsed:.2f}s")
        return flattened
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[SortCenters] ✗ Error after {elapsed:.2f}s: {str(e)[:200]}")
        return []
    finally:
        gremlin_client.close()

def fetch_all_deliveries():
    """Fetch ALL delivery nodes (no time filter)"""
    print(f"[Deliveries] Starting fetch (no time filter)...")
    gremlin_client = get_neptune_client()
    start_time = time.time()
    
    try:
        query = """
        g.V().hasLabel('Delivery')
         .project('vertex_id', 'properties')
         .by(id())
         .by(valueMap())
        """
        
        callback = gremlin_client.submit(query)
        results = callback.all().result()
        
        # Flatten
        flattened = []
        for node in results:
            flat_node = {'vertex_id': node['vertex_id']}
            if node.get('properties'):
                for key, value in node['properties'].items():
                    flat_node[key] = flatten_value(value)
            flattened.append(flat_node)
        
        elapsed = time.time() - start_time
        print(f"[Deliveries] ✓ Fetched {len(flattened):,} records in {elapsed:.2f}s")
        return flattened
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Deliveries] ✗ Error after {elapsed:.2f}s: {str(e)[:200]}")
        return []
    finally:
        gremlin_client.close()

# ===================== EDGE FETCH FUNCTIONS =====================

def fetch_edges_for_packages(edge_type, package_vertex_ids, batch_num, total_batches):
    """Fetch edges filtered by package vertex IDs (with source/destination info)"""
    print(f"[{edge_type}] Batch {batch_num}/{total_batches}: Fetching edges for {len(package_vertex_ids)} packages...")
    gremlin_client = get_neptune_client()
    start_time = time.time()
    
    try:
        vertex_id_list = ','.join([f"'{vid}'" for vid in package_vertex_ids])
        
        query = f"""
        g.V({vertex_id_list})
         .outE('{edge_type}')
         .project('edge_id', 'from_vertex', 'to_vertex', 'properties')
         .by(id())
         .by(outV().id())
         .by(inV().id())
         .by(valueMap())
        """
        
        callback = gremlin_client.submit(query)
        results = callback.all().result()
        
        # Flatten the structure
        flattened_results = []
        for edge in results:
            flat_edge = {
                'edge_id': edge['edge_id'],
                'from_vertex': edge['from_vertex'],
                'to_vertex': edge['to_vertex']
            }
            # Merge properties
            if edge.get('properties'):
                for key, value in edge['properties'].items():
                    flattened = flatten_value(value)
                    # Special handling for container_problems in Problem edges
                    if edge_type == 'Problem' and key == 'container_problems':
                        flat_edge[key] = clean_container_problems(flattened)
                    else:
                        flat_edge[key] = flattened
            flattened_results.append(flat_edge)
        
        elapsed = time.time() - start_time
        print(f"[{edge_type}] Batch {batch_num}/{total_batches}: ✓ Fetched {len(flattened_results):,} edges in {elapsed:.2f}s")
        return flattened_results
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{edge_type}] Batch {batch_num}/{total_batches}: ✗ Error after {elapsed:.2f}s: {str(e)[:200]}")
        return []
    finally:
        gremlin_client.close()

def fetch_all_edges_for_packages_parallel(edge_type, package_vertex_ids, max_workers):
    """Fetch all edges for packages in parallel batches (THREAD-SAFE)"""
    print(f"\n[{edge_type}] Fetching edges for {len(package_vertex_ids):,} packages...")
    
    if not package_vertex_ids:
        print(f"[{edge_type}] ⚠️  No packages to fetch edges for")
        return []
    
    batches = [package_vertex_ids[i:i + PACKAGE_BATCH_SIZE] 
               for i in range(0, len(package_vertex_ids), PACKAGE_BATCH_SIZE)]
    
    total_batches = len(batches)
    print(f"[{edge_type}] Split into {total_batches} batches of max {PACKAGE_BATCH_SIZE} packages each")
    
    all_edges = []
    edges_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(fetch_edges_for_packages, edge_type, batch, i+1, total_batches)
            for i, batch in enumerate(batches)
        ]
        
        for future in as_completed(futures):
            try:
                batch_edges = future.result()
                
                with edges_lock:
                    all_edges.extend(batch_edges)
                    
            except Exception as e:
                print(f"[{edge_type}] ✗ Batch failed: {e}")
    
    print(f"[{edge_type}] ✓ Total edges fetched: {len(all_edges):,}")
    return all_edges

# ===================== FILE OPERATIONS =====================

def clear_local_folder(folder_path):
    """Delete all files in the local folder"""
    print(f"\nClearing folder: {folder_path}")
    try:
        if os.path.exists(folder_path):
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            if file_count > 0:
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                print(f"✓ Deleted {file_count} file(s)")
        else:
            os.makedirs(folder_path, exist_ok=True)
            print(f"✓ Created folder")
    except Exception as e:
        print(f"Note: {str(e)}")
        os.makedirs(folder_path, exist_ok=True)

def write_csv_to_local(data, folder_path, filename):
    """Write DataFrame to local file as CSV with proper quoting"""
    if not data:
        print(f"⚠ No data for {filename}")
        return False
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        df = pd.DataFrame(data)
        file_path = os.path.join(folder_path, filename)
        
        # Quote all fields to handle commas and special characters
        df.to_csv(file_path, index=False, quoting=1, doublequote=True)
        
        file_size = os.path.getsize(file_path)
        print(f"✓ Saved {filename}: {len(data):,} rows ({file_size/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to save {filename}: {str(e)}")
        return False

# ===================== MAIN EXECUTION =====================

def main():
    """Main execution - fetch raw data only"""
    print("="*70)
    print("NEPTUNE GRAPH EXPORT - RAW DATA WITH CLEAN FORMATTING")
    print("="*70)
    print(f"Training range:    {TRAINING_START_DATE} <= delivered_date <= {TRAINING_END_DATE}")
    print(f"Validation range:  {VALIDATION_START_DATE} <= delivered_date <= {VALIDATION_END_DATE}")
    print(f"Test range:        {TEST_START_DATE} <= delivered_date <= {TEST_END_DATE}")
    print(f"Package batch size: {PACKAGE_FETCH_BATCH_SIZE}")
    print(f"Edge batch size:    {PACKAGE_BATCH_SIZE} packages per edge query")
    print(f"Max workers:        {MAX_WORKERS}")
    print(f"Local Directory:    {LOCAL_DIR}")
    print()
    print("FEATURES:")
    print("  ✅ Filter out returns (is_return=true)")
    print("  ✅ Flatten list values (remove [])")
    print("  ✅ Convert plan_route to -> separated format")
    print("  ✅ Convert container_problems to space-separated format")
    print("  ✅ Proper CSV quoting for special characters")
    print("="*70)
    
    overall_start = time.time()
    
    try:
        # Phase 0: Clear folder
        print("\n" + "="*70)
        print("PHASE 0: PREPARATION")
        print("="*70)
        clear_local_folder(LOCAL_DIR)
        
        # Phase 1: Fetch shared nodes
        print("\n" + "="*70)
        print("PHASE 1: FETCH SHARED NODES")
        print("="*70)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_sort_centers = executor.submit(fetch_all_sort_centers)
            future_deliveries = executor.submit(fetch_all_deliveries)
            
            sort_centers = future_sort_centers.result()
            deliveries = future_deliveries.result()
        
        # Phase 2: Fetch ALL packages in PARALLEL (excluding returns)
        print("\n" + "="*70)
        print("PHASE 2: FETCH PACKAGES (EXCLUDING RETURNS)")
        print("="*70)
        
        datasets_config = [
            {'name': 'training', 'start_date': TRAINING_START_DATE, 'end_date': TRAINING_END_DATE},
            {'name': 'validation', 'start_date': VALIDATION_START_DATE, 'end_date': VALIDATION_END_DATE},
            {'name': 'test', 'start_date': TEST_START_DATE, 'end_date': TEST_END_DATE}
        ]
        
        organized_packages = fetch_all_packages_parallel(
            datasets_config,
            batch_size=PACKAGE_FETCH_BATCH_SIZE,
            max_workers=MAX_WORKERS
        )
        
        print("\n" + "="*70)
        print("COMBINING ALL PACKAGES")
        print("="*70)
        
        training_packages = organized_packages.get('training', [])
        validation_packages = organized_packages.get('validation', [])
        test_packages = organized_packages.get('test', [])
        
        all_packages = training_packages + validation_packages + test_packages
        
        print(f"Total packages combined: {len(all_packages):,}")
        print(f"  Training:   {len(training_packages):,}")
        print(f"  Validation: {len(validation_packages):,}")
        print(f"  Test:       {len(test_packages):,}")
        
        if not all_packages:
            print("❌ No packages found. Exiting.")
            return
        
        # Double-check no returns leaked through
        print("\n[Return Filter] Verifying no return packages...")
        return_count = sum(1 for pkg in all_packages if pkg.get('is_return') in [True, 'True', 'true'])
        if return_count > 0:
            print(f"[Return Filter] ⚠️  Found {return_count} return packages, filtering them out...")
            all_packages = [pkg for pkg in all_packages if pkg.get('is_return') not in [True, 'True', 'true']]
            training_packages = [pkg for pkg in training_packages if pkg.get('is_return') not in [True, 'True', 'true']]
            validation_packages = [pkg for pkg in validation_packages if pkg.get('is_return') not in [True, 'True', 'true']]
            test_packages = [pkg for pkg in test_packages if pkg.get('is_return') not in [True, 'True', 'true']]
            print(f"[Return Filter] ✓ Filtered out {return_count} return packages")
        else:
            print(f"[Return Filter] ✓ Confirmed: 0 return packages")
        
        # Get vertex IDs
        all_package_vertex_ids = [pkg['vertex_id'] for pkg in all_packages]
        
        # Phase 3: Fetch edges
        print("\n" + "="*70)
        print("PHASE 3: FETCH EDGES")
        print("="*70)
        
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_induct = executor.submit(fetch_all_edges_for_packages_parallel, 'Induct', all_package_vertex_ids, MAX_WORKERS)
            future_exit202 = executor.submit(fetch_all_edges_for_packages_parallel, 'Exit202', all_package_vertex_ids, MAX_WORKERS)
            future_problem = executor.submit(fetch_all_edges_for_packages_parallel, 'Problem', all_package_vertex_ids, MAX_WORKERS)
            future_missort = executor.submit(fetch_all_edges_for_packages_parallel, 'Missort', all_package_vertex_ids, MAX_WORKERS)
            future_linehaul = executor.submit(fetch_all_edges_for_packages_parallel, 'LineHaul', all_package_vertex_ids, MAX_WORKERS)
            future_delivery = executor.submit(fetch_all_edges_for_packages_parallel, 'Delivery', all_package_vertex_ids, MAX_WORKERS)
            
            induct_edges = future_induct.result()
            exit202_edges = future_exit202.result()
            problem_edges = future_problem.result()
            missort_edges = future_missort.result()
            linehaul_edges = future_linehaul.result()
            delivery_edges = future_delivery.result()
        
        # Phase 4: Save files
        print("\n" + "="*70)
        print("PHASE 4: SAVE FILES")
        print("="*70)
        
        write_csv_to_local(sort_centers, LOCAL_DIR, "sort_centers.csv")
        write_csv_to_local(deliveries, LOCAL_DIR, "deliveries.csv")
        write_csv_to_local(all_packages, LOCAL_DIR, "packages.csv")
        write_csv_to_local(induct_edges, LOCAL_DIR, "induct_edges.csv")
        write_csv_to_local(exit202_edges, LOCAL_DIR, "exit202_edges.csv")
        write_csv_to_local(problem_edges, LOCAL_DIR, "problem_edges.csv")
        write_csv_to_local(missort_edges, LOCAL_DIR, "missort_edges.csv")
        write_csv_to_local(linehaul_edges, LOCAL_DIR, "linehaul_edges.csv")
        write_csv_to_local(delivery_edges, LOCAL_DIR, "delivery_edges.csv")
        
        summary_data = [{
            'export_date': datetime.now().isoformat(),
            'training_start': TRAINING_START_DATE,
            'training_end': TRAINING_END_DATE,
            'validation_start': VALIDATION_START_DATE,
            'validation_end': VALIDATION_END_DATE,
            'test_start': TEST_START_DATE,
            'test_end': TEST_END_DATE,
            'total_packages': len(all_packages),
            'training_packages': len(training_packages),
            'validation_packages': len(validation_packages),
            'test_packages': len(test_packages),
            'filter_returns': 'true',
            'plan_route_format': 'arrow_separated',
            'container_problems_format': 'space_separated',
            'csv_quoting': 'all_fields'
        }]
        
        write_csv_to_local(summary_data, LOCAL_DIR, "_export_summary.csv")
        
        overall_elapsed = time.time() - overall_start
        
        print("\n" + "="*70)
        print("✓ EXPORT COMPLETED")
        print("="*70)
        print(f"\nTotal time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} minutes)")
        print(f"\nData transformations applied:")
        print(f"  ✅ Filtered out return packages (is_return=true)")
        print(f"  ✅ Flattened list values (removed [])")
        print(f"  ✅ Converted datetime to ISO format")
        print(f"  ✅ Converted plan_route: JSON array -> arrow separated (->)")
        print(f"  ✅ Converted container_problems: JSON array -> space separated")
        print(f"  ✅ Applied proper CSV quoting for all fields")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()