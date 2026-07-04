# =============================================================================
# utils.py
# Project-wide utilities: directory management, logging prefixes.
# =============================================================================

import os
from datetime import datetime


def ensure_dir(path):
    """
    Ensure that a directory exists; create it if it doesn't.
    
    Parameters
    ----------
    path : str   directory path
    
    Returns
    -------
    str   same path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def figure_path(filename, fig_dir="figures"):
    """Return full path <fig_dir>/<filename>, creating fig_dir if needed."""
    ensure_dir(fig_dir)
    return os.path.join(fig_dir, filename)


def data_path(filename, data_dir="data"):
    """Return full path <data_dir>/<filename>, creating data_dir if needed."""
    ensure_dir(data_dir)
    return os.path.join(data_dir, filename)


def timestamp():
    """Return ISO-like string for log stamps."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")