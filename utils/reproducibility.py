# ABOUTME: Reproducibility utilities for experiment tracking and state validation.
# ABOUTME: Provides git/DVC state checks and reproduce command generation.

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

SKIP_STAGE_REPRO_CHECK_ENV = "WSI_SKIP_STAGE_REPRO_CHECK"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in TRUTHY_ENV_VALUES


def check_git_clean() -> Tuple[bool, Dict]:
    """
    Check if git state has uncommitted changes.

    Returns:
        tuple: (is_clean, details_dict)
            - is_clean (bool): True if git state is clean
            - details_dict (dict): Details about git state including commit hash
    """
    details = {}

    try:
        # Get current commit hash
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        details["commit_hash"] = git_hash

        # Check git status
        git_status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .decode("utf-8")
            .strip()
        )

        is_clean = not git_status
        details["is_clean"] = is_clean
        details["status"] = git_status.split("\n") if git_status else []

        return is_clean, details

    except subprocess.CalledProcessError as e:
        details["error"] = f"Git command failed: {e}"
        return False, details
    except FileNotFoundError:
        details["error"] = "Git not available"
        return False, details


def check_dvc_clean() -> Tuple[bool, Dict]:
    """
    Check if DVC state has uncommitted changes using dvc diff.

    Returns:
        tuple: (is_clean, details_dict)
            - is_clean (bool): True if DVC state is clean
            - details_dict (dict): Details about DVC state
    """
    details = {}

    # Derive dvc path from the same bin directory as the running Python
    python_bin_dir = os.path.dirname(sys.executable)
    dvc_path = os.path.join(python_bin_dir, "dvc")

    # Fallback to bare 'dvc' if not found in Python's bin dir
    if not os.path.exists(dvc_path):
        dvc_path = "dvc"

    try:
        # Use dvc diff - returns empty when clean, shows changes when dirty
        dvc_diff = subprocess.check_output([dvc_path, "diff"]).decode("utf-8").strip()

        is_clean = not dvc_diff
        details["is_clean"] = is_clean
        details["status"] = dvc_diff.split("\n") if dvc_diff else ["Up to date"]

        return is_clean, details

    except subprocess.CalledProcessError as e:
        details["error"] = f"DVC command failed: {e}"
        return False, details
    except FileNotFoundError:
        # DVC not installed - treat as clean (optional dependency)
        details["warning"] = "DVC not available"
        details["is_clean"] = True
        return True, details


def should_skip_reproducibility_check(input_paths: List[str]) -> bool:
    """
    Check if reproducibility check should be skipped based on input paths.

    Skips check if any input path contains 'demo' or 'test' (case-insensitive).

    Args:
        input_paths: List of input file/directory paths

    Returns:
        True if check should be skipped
    """
    for p in input_paths:
        p_lower = p.lower()
        if "demo" in p_lower or "test" in p_lower:
            return True
    return False


def log_unclean_state(git_details: Dict, dvc_details: Optional[Dict] = None) -> None:
    """
    Log details about unclean git/DVC state to stderr.

    Args:
        git_details: Details dict from check_git_clean()
        dvc_details: Optional details dict from check_dvc_clean()
    """
    print("\n" + "=" * 60, file=sys.stderr)
    print("ERROR: Cannot run with uncommitted changes", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if not git_details.get("is_clean", True):
        print("\nGit has uncommitted changes:", file=sys.stderr)
        for line in git_details.get("status", []):
            if line:
                print(f"  {line}", file=sys.stderr)

    if dvc_details and not dvc_details.get("is_clean", True):
        print("\nDVC has uncommitted changes:", file=sys.stderr)
        for line in dvc_details.get("status", []):
            if line:
                print(f"  {line}", file=sys.stderr)

    print("\nPlease commit or stash your changes before running.", file=sys.stderr)
    print("Or use paths containing 'demo' or 'test' to bypass this check.", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)


def require_clean_state(input_paths: List[str], skip_dvc_check: bool = False) -> Dict:
    """
    Require clean git and DVC state, or exit.

    Skips check if any input path contains 'demo' or 'test'.
    Returns state details for use in reproduce.txt.

    Args:
        input_paths: List of input file/directory paths to check for bypass

    Returns:
        Dict with git_hash and state details (for reproduce.txt generation)

    Raises:
        SystemExit: If state is not clean and bypass not triggered
    """
    if _env_flag_enabled(SKIP_STAGE_REPRO_CHECK_ENV):
        git_clean, git_details = check_git_clean()
        return {
            "bypassed": True,
            "reason": f"Disabled via ${SKIP_STAGE_REPRO_CHECK_ENV}",
            "git_hash": git_details.get("commit_hash", "unknown"),
            "git_clean": git_clean,
        }

    # Check for bypass
    if should_skip_reproducibility_check(input_paths):
        git_clean, git_details = check_git_clean()
        return {
            "bypassed": True,
            "reason": "Input path contains 'demo' or 'test'",
            "git_hash": git_details.get("commit_hash", "unknown"),
            "git_clean": git_clean,
        }

    # Check git state
    git_clean, git_details = check_git_clean()

    # Check DVC state (optional)
    if skip_dvc_check:
        dvc_clean = True
        dvc_details = {"warning": "DVC check skipped"}
    else:
        dvc_clean, dvc_details = check_dvc_clean()

    if not git_clean or not dvc_clean:
        log_unclean_state(git_details, dvc_details)
        sys.exit(1)

    return {
        "bypassed": False,
        "git_hash": git_details.get("commit_hash", "unknown"),
        "git_clean": git_clean,
        "dvc_clean": dvc_clean,
    }


def create_reproduce_command(
    parser: argparse.ArgumentParser,
    output_file: str,
    dvc_files: Optional[List[str]] = None,
    git_hash: Optional[str] = None,
) -> None:
    """
    Create a text file with commands to reproduce this run.

    Args:
        parser: ArgumentParser object (to extract parsed args)
        output_file: File path to save reproduction command
        dvc_files: Optional list of DVC file paths to checkout
        git_hash: Optional git hash (fetched if not provided)
    """
    # Get git hash if not provided
    if git_hash is None:
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash = "unknown"

    # Build reproduce file content
    lines = ["# Reproduce this run", f"git checkout {git_hash}"]

    # Add DVC checkout commands
    if dvc_files:
        for dvc_file in dvc_files:
            lines.append(f"dvc checkout {dvc_file}")
    else:
        # Default DVC files for this project
        lines.append("dvc checkout")

    lines.append("")  # Blank line before command

    # Build the python command
    command_parts = ["python", sys.argv[0]]

    # Parse arguments
    args = parser.parse_args()

    # Create mapping of destinations to default values
    default_values = {}
    for action in parser._actions:
        default_values[action.dest] = action.default

    # Identify store_true/store_false arguments
    store_action_dests = set()
    for action in parser._actions:
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            store_action_dests.add(action.dest)

    # Get option strings for each dest
    dest_to_option = {}
    for action in parser._actions:
        if action.option_strings:
            # Prefer long option (--foo) over short (-f)
            long_opts = [o for o in action.option_strings if o.startswith("--")]
            if long_opts:
                dest_to_option[action.dest] = long_opts[0]
            else:
                dest_to_option[action.dest] = action.option_strings[0]

    # Add non-default args to command
    for arg_name, arg_value in vars(args).items():
        # Skip help
        if arg_name == "help":
            continue

        # Skip if value is the default
        if arg_value == default_values.get(arg_name):
            continue

        # Skip None values
        if arg_value is None:
            continue

        option_str = dest_to_option.get(arg_name)
        if not option_str:
            # Positional argument
            command_parts.append(str(arg_value))
        elif arg_name in store_action_dests:
            # Boolean flag - only add if True and different from default
            if arg_value:
                command_parts.append(option_str)
        elif isinstance(arg_value, list):
            # List argument
            for v in arg_value:
                command_parts.append(f"{option_str} {v}")
        else:
            # Regular argument
            command_parts.append(f"{option_str} {arg_value}")

    lines.append(" \\\n    ".join(command_parts))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
