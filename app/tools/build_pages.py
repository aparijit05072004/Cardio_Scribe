#!/usr/bin/env python3
"""Build the static site published to GitHub Pages.

GitHub Pages serves files, not processes: no Python, no GPU, no model. What it
CAN serve is demo mode, which never needed any of those — it is pure signal
processing, and `app/web/standalone.js` reimplements it in the browser. The
published page therefore does real work (WFDB parsing, the 0.5-40 Hz zero-phase
band-pass, R-peak detection, heart rate and RR variability, all verified
bit-identical to the server) and withholds everything that needs the model.

    python app/tools/build_pages.py --out site

The output is the same `app/web/` files, laid out so that relative asset paths
resolve under a project sub-path such as /cardio_scribe/.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

APP_WEB = Path(__file__).resolve().parent.parent / "web"
ROOT_FILES = ["index.html"]
STATIC_FILES = ["app.js", "styles.css", "standalone.js"]
STATIC_DIRS = ["sample"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="site", help="output directory (default: site)")
    ap.add_argument("--clean", action="store_true", help="remove the output directory first")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    if args.clean and out.exists():
        shutil.rmtree(out)
    (out / "static").mkdir(parents=True, exist_ok=True)

    copied = []
    for name in ROOT_FILES:
        html = (APP_WEB / name).read_text(encoding="utf-8")
        # Tell the page there is no API here, so it goes straight to the
        # in-browser engine instead of probing and logging a 404.
        marker = '<script src="static/standalone.js"></script>'
        if marker not in html:
            raise SystemExit(f"error: {name} no longer loads standalone.js; build aborted")
        html = html.replace(
            marker,
            "<script>window.CARDIOSCRIBE_STATIC = true;</script>\n" + marker,
            1,
        )
        (out / name).write_text(html, encoding="utf-8")
        copied.append(name)
    for name in STATIC_FILES:
        shutil.copy2(APP_WEB / name, out / "static" / name)
        copied.append(f"static/{name}")
    for name in STATIC_DIRS:
        # Source lives at app/web/<name>/ because the server mounts app/web at
        # /static; the built site needs it one level deeper, under static/.
        src = APP_WEB / name
        if src.is_dir():
            shutil.copytree(src, out / "static" / name, dirs_exist_ok=True)
            copied.append(f"static/{name}/")

    # Pages runs Jekyll by default, which would skip files beginning with "_".
    (out / ".nojekyll").write_text("")
    copied.append(".nojekyll")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Built {out}")
    for name in copied:
        print(f"  {name}")
    print(f"  total {total / 1024:.0f} KB")
    print()
    print("Publish: Settings -> Pages -> Source: 'Deploy from a branch',")
    print("         Branch: gh-pages, folder: / (root).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
