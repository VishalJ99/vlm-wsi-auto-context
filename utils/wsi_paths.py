# ABOUTME: WSI path validation utility. Validates that WSI paths exist and returns absolute paths.
# ABOUTME: Requires full paths (no manifest-based filename auto-resolution).
"""
WSI Path Validation Utility

Validates WSI file paths and returns absolute paths.
Users must provide full paths (absolute or relative) to WSI files.
"""

import os


def resolve_wsi_path(wsi_input: str, manifest_path: str = None) -> str:
    """
    Validate WSI path and return absolute path.

    Args:
        wsi_input: Full path to WSI file (absolute or relative)
        manifest_path: Deprecated, ignored. Kept for backward compatibility.

    Returns:
        Absolute path to WSI file

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if os.path.exists(wsi_input):
        return os.path.abspath(wsi_input)

    raise FileNotFoundError(
        f"WSI file not found: {wsi_input}\n"
        f"  Please provide a full path (absolute or relative) to an existing WSI file."
    )


def resolve_wsi_path_or_exit(wsi_input: str, manifest_path: str = None) -> str:
    """
    Validate WSI path, printing error and exiting on failure.

    Convenience wrapper for CLI scripts.
    """
    import sys

    try:
        return resolve_wsi_path(wsi_input, manifest_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
