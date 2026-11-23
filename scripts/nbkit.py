"""Tiny helper to write notebooks from Python lists: md("...") and py("...") cells."""
import textwrap
from pathlib import Path

import nbformat

SETUP = '''# Setup: find the workshop folder and import the helpers (works locally and in Colab)
import os, sys
from pathlib import Path
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents, Path("/content/decision-models-workshop")]
             if (p / "dmw").is_dir()), None)
assert ROOT, "Open this notebook from the workshop folder. In Colab, run 00_setup.ipynb first."
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from dmw import config
config.load_env()'''


def md(s):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip("\n"))


def py(s):
    return nbformat.v4.new_code_cell(textwrap.dedent(s).strip("\n"))


def write(path, cells, title=None):
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if title:
        nb.metadata["title"] = title
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, str(path))
