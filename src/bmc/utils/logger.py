import logging
import psutil
import os
from contextlib import contextmanager
import time

class ResourceProfiler:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_filepath = os.path.join(log_dir, 'resource_usage.log')
        
        # Create a completely isolated logger
        self.resource_logger = logging.getLogger("resource_tracker")
        self.resource_logger.setLevel(logging.INFO)
        
        # CRITICAL: Prevent the root logger from hijacking our messages
        self.resource_logger.propagate = False  
        
        # Bind the file handler
        if not self.resource_logger.handlers:
            self.file_handler = logging.FileHandler(self.log_filepath, mode='a')
            # Updated formatter to give us a bit more space for the new metrics
            formatter = logging.Formatter('%(asctime)s | PID: %(process)d | %(message)s', datefmt='%H:%M:%S')
            self.file_handler.setFormatter(formatter)
            self.resource_logger.addHandler(self.file_handler)

    def log_usage(self, milestone: str):
        """Measures hardware (including threads/cores) and forces a disk write."""
        try:
            process = psutil.Process(os.getpid())
            
            # --- Memory & CPU Load ---
            cpu_percent = psutil.cpu_percent(interval=0.1)
            rss_mem_gb = process.memory_info().rss / (1024 ** 3)
            sys_mem_percent = psutil.virtual_memory().percent
            
            # --- Threads & Cores ---
            # How many threads are currently active inside this specific script?
            num_threads = process.num_threads()
            
            # How many CPU cores is the OS allowing this script to use?
            try:
                # cpu_affinity works on Windows and Linux, showing explicitly allowed cores
                usable_cores = len(process.cpu_affinity())
            except AttributeError:
                # Fallback for environments like macOS where affinity isn't supported
                usable_cores = psutil.cpu_count(logical=True)
            
            # Log the upgraded metrics
            self.resource_logger.info(
                f"[{milestone}] - CPU Load: {cpu_percent}% ({usable_cores} Cores) | "
                f"Active Threads: {num_threads} | Script RAM: {rss_mem_gb:.2f} GB | Server RAM: {sys_mem_percent}%"
            )
            
            # CRITICAL: Force the OS to bypass buffer and write to the text file instantly
            for handler in self.resource_logger.handlers:
                handler.flush()
                
        except Exception as e:
            print(f"RESOURCE LOGGER FAILED: {e}")

    @contextmanager
    def track_strain(self, task_name: str):
        """
        Acts as a stopwatch to measure the exact time, CPU, and RAM 
        strain caused by a specific function or block of code.
        """
        process = psutil.Process(os.getpid())
        
        # 1. Take the baseline snapshots
        start_time = time.time()
        start_ram = process.memory_info().rss / (1024 ** 3)
        
        # Calling cpu_percent with interval=None resets the psutil CPU counter
        psutil.cpu_percent(interval=None) 
        
        try:
            # 2. Let the wrapped function run!
            yield 
            
        finally:
            # 3. Take the post-execution snapshots
            end_time = time.time()
            end_ram = process.memory_info().rss / (1024 ** 3)
            
            # Calling it again returns the average CPU usage since the first call
            avg_cpu = psutil.cpu_percent(interval=None) 
            
            # 4. Calculate the exact strain (Delta)
            duration = end_time - start_time
            ram_diff = end_ram - start_ram
            
            # Format with + or - signs for easy reading
            self.resource_logger.info(
                f"[STRAIN] {task_name} | Duration: {duration:.1f}s | "
                f"Avg CPU: {avg_cpu}% | RAM Shift: {ram_diff:+.2f} GB | Final RAM: {end_ram:.2f} GB"
            )
            
            for handler in self.resource_logger.handlers:
                handler.flush()

def log_execution(logger, message, level=logging.WARNING, **kwargs):
    """
    Helper function to log in case a logger is provided. Falls back to print.
    Accepts kwargs (like exc_info=True) to pass directly to the logger.

    Parameters
    ----------

    logger : logging.Logger
        A logger object
    message : str
        The string that is to be recorded by the logging function 

    See Also
    --------

    src.cube.spatiotemporal.spatiotemporal_cube._setup_pipeline_logger
    """
    if logger:
        logger.log(level, message, **kwargs)
    # Setup safeguard to print message when logger is None
    else:
        print(message)