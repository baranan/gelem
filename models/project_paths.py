"""
models/project_paths.py

ProjectPaths -- a frozen, Qt-free value object naming where THIS project's
files live: its root folder, its artifacts cache, and where operators write
their outputs. See docs/architecture.md section 2 ("Two more things exist
but are not components") and section 6 (the `run` object).

Replaces the construction-time directory bag that operators/operator_config.py
used to hand each operator factory (since deleted), built once in main.py
from the CODEBASE INSTALL directory -- not the researcher's project -- and
never re-rooted on save or load (docs/review/p1.9-survey.md).

An unsaved project works inside a WORKSPACE folder, laid out identically to
a saved project (an artifacts/ and an outputs/ subfolder under the same
root) so no operator or cache code needs to know which case it is in.
is_workspace only distinguishes the two for a caller that cares -- for
example a future close-time prompt or Save-As migration, neither of which
is this item.

AppController holds the current ProjectPaths and replaces it wholesale on
save_project() and load_project(), at the same point it already re-roots
ArtifactStore (docs/media_architecture.md section 4.7). build_project_paths
below is pure -- it only computes paths -- so the controller does nothing
but swap the value; this module has no opinion on *when* to swap it.
"""

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Where this project's files live.

    root:          the project folder -- a saved project's folder, or an
                    unsaved project's workspace folder.
    artifacts_dir: the derived-image cache, root/"artifacts". Handed to
                    ArtifactStore.set_artifacts_dir() (media_architecture.md
                    section 4.7); this module does not create it -- the
                    store creates its own directory.
    outputs_dir:   root/"outputs". Every operator that writes a file writes
                    under here, in a subfolder it names for itself (e.g.
                    "frames", "plots") -- there is no operator-specific
                    field on ProjectPaths itself.
    is_workspace:  True when root is an unsaved project's workspace rather
                    than a folder the researcher chose with Save.
    """

    root: Path
    artifacts_dir: Path
    outputs_dir: Path
    is_workspace: bool


def build_project_paths(root: Path, *, is_workspace: bool) -> ProjectPaths:
    """Pure builder: every field is derived from root and nothing else.

    Touches no filesystem. A caller that needs the directories to exist
    creates them -- create_workspace() below does, for the workspace case;
    a saved project's root already exists because Dataset.save()/load()
    require it, and artifacts_dir/outputs_dir are created lazily by
    whatever first writes into them (ArtifactStore.set_artifacts_dir(), or
    an operator's own mkdir(parents=True) under outputs_dir).
    """
    root = Path(root)
    return ProjectPaths(
        root=root,
        artifacts_dir=root / "artifacts",
        outputs_dir=root / "outputs",
        is_workspace=is_workspace,
    )


def default_workspaces_root() -> Path:
    """The per-user app-data folder workspaces live under by default --
    NEVER the OS temp directory (a workspace's operator outputs and
    artifact cache must survive the kind of cleanup a temp folder invites).
    Qt-free by construction: QStandardPaths needs a QApplication, and this
    module must not depend on one.

    Always a plain default; every caller that actually builds a workspace
    takes the root as a constructor argument (create_workspace() below)
    rather than reaching in here itself, so a test or a future setting can
    supply its own root without touching this function.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
        return Path(base) / "Gelem" / "workspaces"
    if sys.platform == "darwin":
        return (
            Path.home() / "Library" / "Application Support" / "Gelem"
            / "workspaces"
        )
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "Gelem" / "workspaces"


def create_workspace(workspaces_root: Path) -> ProjectPaths:
    """Creates one new, uniquely named workspace folder under
    workspaces_root and returns its ProjectPaths, with the root already
    created on disk.

    Called once per app launch (main.py). workspaces_root is always a
    caller-supplied argument -- never a hardcoded constant here -- so a
    test can point it at tmp_path. The folder is never deleted by this
    item: workspace cleanup is future work.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{uuid.uuid4().hex[:8]}"
    paths = build_project_paths(Path(workspaces_root) / name, is_workspace=True)
    paths.root.mkdir(parents=True, exist_ok=True)
    return paths
