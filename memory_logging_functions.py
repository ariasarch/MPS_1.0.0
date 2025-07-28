import time
import os
import psutil
import tracemalloc
import gc
import pandas as pd
import numpy as np
import logging
import warnings
import traceback
import json
import sys
from dask.distributed import get_client, get_worker

logger = logging.getLogger(__name__)

def get_initial_file_stats(dpath):
    total_size = 0
    file_count = 0
    file_stats = []
    for root, dirs, files in os.walk(dpath):
        for file in files:
            if file.endswith('.avi'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1
                file_stats.append({
                    "file_name": file,
                    "file_path": file_path,
                    "file_size_bytes": file_size,
                    "file_size_mb": file_size / (1024 * 1024)
                })
    
    summary = {
        "total_size_gb": total_size / (1024**3),
        "file_count": file_count,
        "average_file_size_mb": (total_size / file_count) / (1024**2) if file_count else 0
    }
    
    return summary, file_stats

def export_initial_file_stats(summary, file_stats, save_to_path):
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(save_to_path, 'initial_file_stats_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Initial file statistics summary exported to {summary_path}")

    if file_stats:
        details_df = pd.DataFrame(file_stats)
        details_path = os.path.join(save_to_path, 'initial_file_stats_details.csv')
        details_df.to_csv(details_path, index=False)
        logger.info(f"Detailed initial file statistics exported to {details_path}")
    else:
        logger.warning("No detailed file statistics to export")

def get_worker_stats():
    worker = get_worker()
    process = psutil.Process()
    
    stats = {
        "memory_rss": process.memory_info().rss,
        "memory_vms": process.memory_info().vms,
        "cpu_percent": process.cpu_percent(interval=0.1),
    }
    
    net_io = psutil.net_io_counters()
    stats.update({
        "network_send": net_io.bytes_sent,
        "network_recv": net_io.bytes_recv,
    })
    
    try:
        disk_io = psutil.disk_io_counters()
        stats.update({
            "disk_read": disk_io.read_bytes,
            "disk_write": disk_io.write_bytes,
        })
    except Exception:
        stats.update({
            "disk_read": 0,
            "disk_write": 0,
        })
    
    try:
        connections = process.connections()
        stats["network_connections"] = len(connections)
    except psutil.AccessDenied:
        stats["network_connections"] = "Access Denied"
    except Exception as e:
        stats["network_connections"] = f"Error: {str(e)}"
    
    stats.update({
        "data_size": len(getattr(worker, 'data', {})),
        "nthreads": getattr(worker, 'nthreads', 0),
    })
    
    stats["dask_memory"] = sum(sys.getsizeof(v) for v in worker.data.values())
    
    return stats

def enhanced_log_dask_metrics(client=None):
    if client is None:
        client = get_client()
    
    logger.info("Starting enhanced_log_dask_metrics")
    scheduler_info = client.scheduler_info()
    
    worker_stats = client.run(get_worker_stats)
    logger.info(f"Worker stats: {worker_stats}")
    
    metrics = {
        "timestamp": time.time(),
        "total_worker_memory_gb": sum(w.get('memory_rss', 0) for w in worker_stats.values()) / (1024**3),
        "number_of_workers": len(scheduler_info.get('workers', {})),
    }
    
    processing_tasks = client.processing()
    has_what_tasks = client.has_what()
    all_tasks = {**processing_tasks, **has_what_tasks}
    
    tasks_completed = sum(len(tasks) for tasks in has_what_tasks.values())
    tasks_processing = sum(len(tasks) for tasks in processing_tasks.values())
    tasks_pending = len(client.who_has()) - tasks_completed
    
    metrics.update({
        "tasks_completed": tasks_completed,
        "tasks_processing": tasks_processing,
        "tasks_pending": tasks_pending,
    })
    
    for stat in ['network_send', 'network_recv', 'disk_read', 'disk_write']:
        metrics[stat] = sum(w.get(stat, 0) for w in worker_stats.values())
    
    metrics["avg_cpu_percent"] = np.mean([w.get('cpu_percent', 0) for w in worker_stats.values()])
    
    network_connections = [w.get('network_connections', 'N/A') for w in worker_stats.values()]
    metrics["total_network_connections"] = sum(nc for nc in network_connections if isinstance(nc, int))
    
    metrics["total_rss_memory_gb"] = sum(w.get('memory_rss', 0) for w in worker_stats.values()) / (1024**3)
    metrics["total_vms_memory_gb"] = sum(w.get('memory_vms', 0) for w in worker_stats.values()) / (1024**3)
    metrics["total_dask_memory_gb"] = sum(w.get('dask_memory', 0) for w in worker_stats.values()) / (1024**3)
    
    metrics["worker_stats"] = worker_stats
    
    return metrics

def export_dask_metrics(dask_logs, save_to_path):
    logger.info("Starting export_dask_metrics")
    logger.info(f"Number of log entries: {len(dask_logs)}")
    logger.info(f"First log entry: {dask_logs[0] if dask_logs else 'No logs'}")
    
    if not dask_logs:
        logger.warning("No Dask logs to export")
        return
    
    try:
        dask_df = pd.DataFrame(dask_logs)
        logger.info(f"DataFrame shape: {dask_df.shape}")
        logger.info(f"DataFrame columns: {dask_df.columns}")
        
        save_path = os.path.join(save_to_path, 'dask_metrics.csv')
        logger.info(f"Saving to: {save_path}")
        dask_df.to_csv(save_path, index=False, mode='a', header=not os.path.exists(save_path))
        logger.info(f"Dask metrics exported to {save_path}")
    except Exception as e:
        logger.error(f"Error exporting Dask metrics: {e}")
        logger.error(traceback.format_exc())

    json_path = os.path.join(save_to_path, 'dask_metrics_backup.json')
    with open(json_path, 'w') as f:
        json.dump(dask_logs, f)
    logger.info(f"Dask metrics backup saved to {json_path}")

def log_intermediate_results(data):
    logger.info(f"Intermediate results for {data.name}:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Dtype: {data.dtype}")
    logger.info(f"  Coordinates: {data.coords}")
    logger.info(f"  Attributes: {data.attrs}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_val = data.mean().compute().item()
        std_val = data.std().compute().item()
        min_val = data.min().compute().item()
        max_val = data.max().compute().item()
    
    logger.info(f"  Mean: {mean_val}")
    logger.info(f"  Std Dev: {std_val}")
    logger.info(f"  Min: {min_val}")
    logger.info(f"  Max: {max_val}")
    
    sample_values = data.isel(frame=slice(0, 5)).compute().values.flatten()
    logger.info(f"  First few values: {sample_values}")

class MemoryBenchmark:
    def __init__(self):
        self.all_stats = []

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            tracemalloc.start()
            start_time = time.time()
            process = psutil.Process()
            start_process_memory = process.memory_info()
            start_system_memory = psutil.virtual_memory()
            start_swap_memory = psutil.swap_memory()

            result = func(*args, **kwargs)

            end_time = time.time()
            end_process_memory = process.memory_info()
            end_system_memory = psutil.virtual_memory()
            end_swap_memory = psutil.swap_memory()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            process_memory_change = end_process_memory.rss - start_process_memory.rss
            system_memory_change = end_system_memory.used - start_system_memory.used
            swap_memory_change = end_swap_memory.used - start_swap_memory.used

            try:
                net_connections = len(process.net_connections())
            except AttributeError:
                net_connections = "Not available"

            stats = {
                "Function": func.__name__,
                "Execution Time (s)": end_time - start_time,
                "Process Memory Change (MB)": process_memory_change / (1024 * 1024),
                "System Memory Change (MB)": system_memory_change / (1024 * 1024),
                "Swap Memory Change (MB)": swap_memory_change / (1024 * 1024),
                "Peak Process Memory (MB)": peak / (1024 * 1024),
                "Final Process Memory (MB)": end_process_memory.rss / (1024 * 1024),
                "Final System Memory Used (MB)": end_system_memory.used / (1024 * 1024),
                "Available System Memory (MB)": end_system_memory.available / (1024 * 1024),
                "Process CPU Time (s)": process.cpu_times().user + process.cpu_times().system,
                "Process Memory Percent": process.memory_percent(),
                "System Memory Percent": end_system_memory.percent,
                "Swap Memory Percent": end_swap_memory.percent,
                "Number of Threads": process.num_threads(),
                "Number of Open Files": len(process.open_files()),
                "Number of Network Connections": net_connections,
                "Garbage Collection Stats": gc.get_stats(),
            }

            self.all_stats.append(stats)

            print(f"\nMemory Statistics for {func.__name__}:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            return result
        return wrapper

    def export_memory_stats(self, save_to_path):
        if self.all_stats:
            print(f"Number of stats collected: {len(self.all_stats)}")
            df = pd.DataFrame(self.all_stats)
            save_path = os.path.join(save_to_path, 'memory_stats.csv')
            df.to_csv(save_path, index=False, mode='a', header=not os.path.exists(save_path))
            print(f"Memory statistics exported to {save_path}")
        else:
            print("No memory statistics collected.")

memory_benchmark = MemoryBenchmark()