# =============================================================================
# utils.py
# =============================================================================

import os


def ensure_dir(path):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def figure_path(filename, fig_dir=None):
    """
    Return full path  <fig_dir>/<filename>,
    creating fig_dir first if needed.
    Imports FIG_DIR from config to avoid circular imports at module level.
    """
    if fig_dir is None:
        from config import FIG_DIR
        fig_dir = FIG_DIR
    ensure_dir(fig_dir)
    return os.path.join(fig_dir, filename)


def data_path(filename, data_dir=None):
    """
    Return full path  <data_dir>/<filename>,
    creating data_dir first if needed.
    """
    if data_dir is None:
        from config import DATA_DIR
        data_dir = DATA_DIR
    ensure_dir(data_dir)
    return os.path.join(data_dir, filename)