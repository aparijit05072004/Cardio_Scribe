#!/usr/bin/env python3
"""Export the notebook's pipeline into an importable `cardioscribe_runtime.py`.

`cardioscribe()` is defined in notebook section 39 and closes over roughly forty
module-level objects built by the cells above it — the encoder, the Q-Former,
the 4-bit LLM, the calibration temperature, the dataset-level standardization
statistics, and `text_emb['train']` / `split_df['train']` for precedent
retrieval. There is no way to lift the function out on its own: the module has
to *run* the notebook's setup, which is exactly what this script produces.

    python app/tools/export_runtime.py --out ./runtime
    CARDIOSCRIBE_RUNTIME_DIR=./runtime python -m backend.runtime_live --self-check

What you get is a starting point that needs reading before you trust it. The
script drops cells it can identify as demonstrations (bare `print`, plotting,
`%pip`), but it cannot know which of your cells are essential, and importing the
result will download models and read PTB-XL. Review it, then run the self-check.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

NOTEBOOK = "CardioScribe_v2.ipynb"

# Which notebook sections belong in an inference runtime.
#   1-41  the pipeline itself, up to the one-call API
#   46-47 deterministic PR/QRS/QT/QTc/axis and the complete number validator
# Sections 42-45 (target-closure scoring) and 48-51 (PTB-XL+ cross-check,
# external validation, gold set, SSL pretraining) are evaluation work: they want
# the whole test fold, not one uploaded record, so they are left out.
DEFAULT_SECTIONS = set(range(0, 42)) | {46, 47}

SECTION_RE = re.compile(r"^#\s*(\d{1,2})[a-z]?[.)]\s", re.M)
MAGIC_RE = re.compile(r"^\s*[%!].*$", re.M)
PLOT_MARKERS = ("plt.show()", "plt.subplots", "plt.figure", "sns.heatmap", "display(")


def defines_something(src: str) -> bool:
    """True if the cell binds a name the rest of the module might need.

    A cell that only draws a figure can go; a cell that draws a figure *and*
    defines UNCERTAIN_THRESHOLD_CALIBRATED cannot, which is precisely the case
    for section 32's risk-coverage curve.
    """
    try:
        tree = ast.parse(MAGIC_RE.sub("", src))
    except SyntaxError:
        return True        # keep anything we cannot parse; the user reviews it
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    return True
    return False


HEADER = '''"""cardioscribe_runtime — GENERATED, review before use.

Produced by app/tools/export_runtime.py from {notebook}.
Importing this module EXECUTES the notebook's setup: it reads PTB-XL, builds the
encoder and Q-Former, loads the 4-bit LLM and restores the checkpoints. Expect
it to take minutes and to occupy GPU memory for the life of the process.

Required before import:
  CARDIOSCRIBE_DATA  — PTB-XL root
  CARDIOSCRIBE_WORK  — directory holding the trained checkpoints
"""
'''


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=None, help=f"path to {NOTEBOOK}")
    ap.add_argument("--out", required=True, help="directory to write cardioscribe_runtime.py into")
    ap.add_argument("--through-section", type=int, default=41,
                    help="last pipeline section to include (default 41, the inference API; "
                         "sections 46-47 are always included for interval measurement)")
    ap.add_argument("--keep-demos", action="store_true",
                    help="keep plotting and demonstration cells")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    nb_path = Path(args.notebook) if args.notebook else repo / NOTEBOOK
    if not nb_path.exists():
        print(f"error: notebook not found at {nb_path}")
        return 1

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cardioscribe_runtime.py"

    sections = set(DEFAULT_SECTIONS)
    if args.through_section != 41:
        sections = set(range(0, args.through_section + 1)) | {46, 47}

    chunks: list[str] = [HEADER.format(notebook=nb_path.name)]
    kept = skipped = 0
    current_section = 0

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if not src.strip():
            continue

        match = SECTION_RE.search(src)
        if match:
            current_section = int(match.group(1))

        if current_section not in sections:
            skipped += 1
            continue
        if (not args.keep_demos
                and any(m in src for m in PLOT_MARKERS)
                and not defines_something(src)):
            skipped += 1
            continue

        src = MAGIC_RE.sub("", src)      # IPython magics are not valid Python
        chunks.append(src.rstrip() + "\n")
        kept += 1

    separator = "\n\n# " + "-" * 70 + "\n\n"
    out_path.write_text(separator.join(chunks), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"  kept {kept} cells, skipped {skipped}")
    print()
    print("Next:")
    print("  1. READ IT. This is generated code that will load models on import.")
    print(f"  2. CARDIOSCRIBE_RUNTIME_DIR={out_dir} python -m backend.runtime_live --self-check")
    print(f"  3. CARDIOSCRIBE_RUNTIME_DIR={out_dir} uvicorn backend.app:app")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
