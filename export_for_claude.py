"""
export_for_claude.py

Creates a single text file containing all source code in the Gelem
project, formatted for pasting into a new Claude session.

The output file gives Claude a complete picture of the codebase:
- Every .py file with its path relative to the project root
- Every .ipynb file (Jupyter notebooks)
- A header explaining what the file is and how to use it

Usage:
    python export_for_claude.py

Output:
    gelem_codebase_for_claude.txt in the project root.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root of the project — the folder containing this script.
ROOT = Path(__file__).parent

# Output file path.
OUTPUT = ROOT / "gelem_codebase_for_claude.txt"

# File extensions to include.
INCLUDE_EXTENSIONS = {".py", ".ipynb", ".yaml", ".json"}

# Folders and files to skip entirely.
SKIP_FOLDERS = {
    ".venv",
    "__pycache__",
    ".git",
    "gelem_project",
    "gelem_artifacts",
    "tests/renderer_output",
}

SKIP_FILES = {
    "export_for_claude.py",        # This script itself.
    "gelem_codebase_for_claude.txt",  # The output file.
}

# ---------------------------------------------------------------------------
# Collect files
# ---------------------------------------------------------------------------

def should_skip(path: Path) -> bool:
    """Returns True if this path should be excluded from the output."""
    # Skip if any part of the path matches a skip folder.
    for part in path.parts:
        if part in SKIP_FOLDERS:
            return True
    # Skip specific files.
    if path.name in SKIP_FILES:
        return True
    return False


def collect_files() -> list[Path]:
    """
    Walks the project tree and returns all files to include,
    sorted by path for consistent ordering.
    """
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in INCLUDE_EXTENSIONS:
            continue
        if should_skip(path):
            continue
        files.append(path)
    return sorted(files)


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------

HEADER = """
================================================================================
GELEM CODEBASE — FOR CLAUDE
================================================================================

This file contains the complete source code of the Gelem project,
a desktop visual data explorer for psychology research built in Python
with PySide6/Qt. 

In its current state, we create code with the purpose that three undergraduate student programmers could easily 
program the first few stages of this project, and a more experienced programmer will continue to 
develop it after the initial version. 

HOW TO USE THIS FILE:
    Paste this entire file into a new Claude session and say:
    "This is the complete source code of my Gelem project.
     Please read it carefully before I ask you questions."

PROJECT STRUCTURE:
    models/         — Dataset and QueryEngine (Student B)
    artifacts/      — ArtifactStore
    column_types/   — ColumnTypeRegistry and render functions (Student C)
    operators/      — Analysis plugins (Student C)
    ui/             — All PySide6 widgets (Student A)
    tests/          — Test scripts
    controller.py   — AppController (central wiring layer)
    main.py         — Application entry point

KEY DESIGN PRINCIPLES:
    - Only Dataset may modify any table.
    - Background threads never call Dataset directly — they emit
      results, AppController applies them on the main thread.
    - The UI never imports pandas, PIL, mediapipe, or cv2.
    - Every operator returns a dict; AppController writes it to Dataset.
    - All column names come from the researcher's data — Gelem makes
      no assumptions about column names beyond row_id, full_path,
      file_name.

================================================================================
""".strip()


def format_file(path: Path) -> str:
    """
    Formats one file's content with a clear header showing its path.
    """
    relative = path.relative_to(ROOT)
    separator = "=" * 80

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")
    except Exception as e:
        content = f"[Could not read file: {e}]"

    return (
        f"\n{separator}\n"
        f"FILE: {relative}\n"
        f"{separator}\n\n"
        f"{content}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    files = collect_files()

    print(f"Collecting {len(files)} files...")

    lines = [HEADER, "\n"]

    # Add a table of contents.
    lines.append("TABLE OF CONTENTS:")
    lines.append("-" * 40)
    for f in files:
        lines.append(f"  {f.relative_to(ROOT)}")
    lines.append("\n")

    # Add each file's content.
    for f in files:
        print(f"  {f.relative_to(ROOT)}")
        lines.append(format_file(f))

    output_text = "\n".join(lines)
    OUTPUT.write_text(output_text, encoding="utf-8")

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"\nDone. Written to: {OUTPUT}")
    print(f"Size: {size_kb:.1f} KB")
    print(f"Files included: {len(files)}")


if __name__ == "__main__":
    main()