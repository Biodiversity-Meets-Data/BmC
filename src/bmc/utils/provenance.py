import os
import re
import json
import platform
import subprocess
import importlib.metadata
import importlib
from datetime import datetime, timezone
import logging
from typing import Optional, List, Set

def _find_repo_root() -> str:
    """Uses git to find the absolute root directory of the repository. Falls back to current dir."""
    try:
        return subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return os.getcwd()

def _discover_project_dependencies(repo_root: str) -> set:
    """
    Scans standard configuration files in the repository root to automatically 
    extract a list of required Python packages, handling various file encodings.
    """
    import re
    import os
    
    packages = set()

    def _safe_read_file(filepath: str) -> list:
        """Safely reads a text file by falling back through common encodings."""
        # utf-16 handles PowerShell exports, utf-8-sig handles standard Windows BOMs
        for encoding in ['utf-8', 'utf-16', 'utf-8-sig', 'cp1252']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    return f.readlines()
            except UnicodeDecodeError:
                continue
        log_execution(None, f"Warning: Could not decode {filepath} with known encodings.", logging.WARNING)
        return []

    # 1. Parse requirements.txt
    req_path = os.path.join(repo_root, "requirements.txt")
    if os.path.exists(req_path):
        for line in _safe_read_file(req_path):
            line = line.strip()
            # Ignore comments, empty lines, and pip flags
            if line and not line.startswith(('#', '-e', '-r', '--')):
                # Strip version pins (==, >=, <=, ~) and extras ([...])
                pkg = re.split(r'[=<>~\[]', line)[0].strip()
                if pkg: packages.add(pkg)

    # 2. Parse environment.yml (Conda)
    env_path = os.path.join(repo_root, "environment.yml")
    if os.path.exists(env_path):
        in_deps = False
        for line in _safe_read_file(env_path):
            line = line.strip()
            if line.startswith("dependencies:"):
                in_deps = True
                continue
            if in_deps:
                # Capture conda packages or pip nested packages
                if line.startswith("-") and not line.startswith("- pip"):
                    pkg = re.split(r'[=<>~ ]', line.strip("- "))[0].strip()
                    if pkg: packages.add(pkg)
                # Stop parsing if we hit a new root-level YAML key
                elif line and not line.startswith("-") and ":" in line:
                    in_deps = False

    # 3. Parse pyproject.toml (Poetry / Modern Pip)
    toml_path = os.path.join(repo_root, "pyproject.toml")
    if os.path.exists(toml_path):
        content = "".join(_safe_read_file(toml_path))
        # Look for dependencies array in standard TOML formats
        deps_blocks = re.findall(r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL)
        for block in deps_blocks:
            # Extract quoted package names
            for match in re.findall(r'["\']([^"\']+)["\']', block):
                pkg = re.split(r'[=<>~\[]', match)[0].strip()
                if pkg: packages.add(pkg)

    # Filter out base python declaration if captured
    packages.discard("python")
    return packages

def generate_provenance_metadata(
    recipe: dict, 
    target_package: str = "bmc", 
    extra_packages: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Generates a comprehensive JSON metadata file capturing the execution environment, 
    hardware specifications, software versions, and configuration recipe.
    """
    base_dir = recipe.get('paths', {}).get('base_dir', './')
    os.makedirs(base_dir, exist_ok=True)
    
    metadata_path = os.path.join(base_dir, "provenance_metadata.json")
    
    # --- 1. Fetch Git Provenance ---
    repo_root = _find_repo_root()

    def get_git_revision_hash() -> str:
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        except Exception:
            return "unknown_or_not_git_repo"

    def get_git_branch() -> str:
        try:
            return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        except Exception:
            return "unknown_branch"

    # --- 2. Auto-Discover & Fetch Software Versions ---
    def get_package_version(pkg_name: str) -> str:
        try:
            return importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            try:
                module = importlib.import_module(pkg_name)
                return getattr(module, '__version__', "unknown_version")
            except ImportError:
                return "Not Installed"

    # Automatically scan the repository for required dependencies!
    packages_to_track = _discover_project_dependencies(repo_root)
    
    if extra_packages:
        packages_to_track.update(extra_packages)
        
    # If the scanner found nothing, fallback to a sensible default
    if not packages_to_track:
        packages_to_track = {'xarray', 'rioxarray', 'rasterio', 'numpy'}

    software_env = {pkg: get_package_version(pkg) for pkg in packages_to_track}
    
    # GDAL C++ backend check
    try:
        import rasterio
        software_env["gdal_cpp_backend"] = rasterio.__gdal_version__
    except ImportError:
        software_env["gdal_cpp_backend"] = "Unknown"

    # --- 3. Fetch Hardware Information ---
    hardware_info = {"cpu_cores": os.cpu_count()}
    try:
        import psutil
        hardware_info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        hardware_info["total_ram_gb"] = "psutil library not installed"

    # --- 4. Compile the full Provenance Dictionary ---
    provenance_record = {
        "cube_name": recipe.get('raw_config', {}).get('cube_name', 'unknown_lake'),
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_provenance": {
            "pipeline_package": target_package,
            "pipeline_version": get_package_version(target_package),
            "git_branch": get_git_branch(),
            "git_commit_sha": get_git_revision_hash()
        },
        "system_environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
        "hardware_environment": hardware_info,
        "software_environment": software_env,
        "execution_recipe": recipe
    }

    # --- 5. Write to Disk ---
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(provenance_record, f, indent=4, ensure_ascii=False)

    if logger:
        logger.info(f"Provenance metadata securely saved to: {metadata_path}")
        logger.info(f"Auto-tracked versions for {len(software_env)} repository dependencies.")

    return metadata_path