# -*- coding: utf-8 -*-
"""Assemble the elsarticle submission file from the content source.

WHY A SCRIPT AND NOT A COPY-PASTE
---------------------------------
The Physica A submission file differs from main_physa.tex in exactly two
places: the front matter (elsarticle class, frontmatter environment, ead /
affiliation syntax) and the back matter (Elsevier declaration headings rather
than MDPI ones).  Everything between them is the same paper.  Reproducing 1300
lines of body text by hand is how a number silently changes, so the body is
lifted mechanically from the content source and the assembly is asserted.

WHAT IT DOES
------------
  1. reads main_physa.tex            (content source, article class)
  2. reads manuscript_physa2026.tex  (elsarticle front matter + a sentinel)
  3. takes everything from \\section{Introduction} up to the declaration block
     and splices it in
  4. writes the result back to manuscript_physa2026.tex

The splice is idempotent: once the sentinel has been consumed by a first run,
later runs replace the previously spliced body region in place and leave the
declaration block (which the authors hand-edit) untouched.

The declaration block is NOT lifted: Elsevier's headings differ from MDPI's and
are written by hand in the template file.

Output: overwrites manuscript_physa2026.tex; prints what it spliced.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "main_physa.tex")            # content source
DST = os.path.join(HERE, "manuscript_physa2026.tex")  # elsarticle submission
SENTINEL = "% __MORE__"
BODY_START = r"\section{Introduction}"
DECL_MARKERS = (r"\vspace{6pt}", r"\bibliographystyle{plain}")


def read(p: str) -> str:
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def main() -> int:
    src = read(SRC)
    dst = read(DST)

    i = src.find(BODY_START)
    if i < 0:
        print(f"FAIL: '{BODY_START}' not found in {os.path.basename(SRC)}")
        return 1
    j = len(src)
    for m in DECL_MARKERS:
        k = src.find(m, i)
        if k >= 0:
            j = min(j, k)
    body = src[i:j].rstrip()

    if SENTINEL in dst:
        out = dst.replace(SENTINEL, body + decl)
    else:
        # Idempotent re-splice: the sentinel was consumed by a previous run.
        # Replace only the previously spliced BODY region, from the start of
        # the introduction to just before the declaration block.  The
        # declaration block itself may have been hand-edited since (its TODOs
        # are filled in by the authors) and must survive every re-run.
        d = dst.find(BODY_START)
        c = dst.find(r"\section*{CRediT authorship contribution statement}")
        if d < 0 or c < 0 or c <= d:
            print(f"FAIL: sentinel {SENTINEL!r} absent and no spliced body "
                  f"region found in {os.path.basename(DST)} -- refusing to "
                  f"overwrite")
            return 1
        out = dst[:d] + body + "\n\n" + dst[c:]
    # NOTE: in the sentinel branch `out` already carries the declaration
    # block, and in the re-splice branch `dst[c:]` already carries it; there
    # is no third assembly step.  An earlier draft overwrote `out` with a
    # sentinel replace AFTER both branches, which silently wrote back the OLD
    # body whenever the sentinel was gone -- re-splices looked successful and
    # changed nothing.  Do not reintroduce that line.

    decl = (
        "\n\n\\section*{CRediT authorship contribution statement}\n"
        "TODO author confirmation of the role split.\n\n"
        "\\section*{Declaration of competing interest}\n"
        "TODO author confirmation.\n\n"
        "\\section*{Funding}\n"
        "TODO author confirmation (state funding or none).\n\n"
        "\\section*{Data availability}\n"
        "TODO before submission---state the repository and DOI for the "
        "protocol modules, the simulation code, the result artefacts, and the "
        "scripts that regenerate every number, table and figure.\n\n"
        "\\section*{Use of AI}\n"
        "TODO author confirmation---state the tool and the scope of its use "
        "honestly.\n\n"
        "\\bibliographystyle{elsarticle-harv}\n"
        "\\bibliography{refs}\n\n"
        "\\end{document}\n"
    )

    with open(DST, "w", encoding="utf-8") as fh:
        fh.write(out)

    n_sec = len(re.findall(r"\\section\{", body))
    n_tab = len(re.findall(r"\\begin\{table\}", body))
    n_fig = len(re.findall(r"\\begin\{figure\}", body))
    n_eq = len(re.findall(r"\\begin\{equation\}", body))
    print(f"spliced {len(body)} chars from {os.path.basename(SRC)}")
    print(f"  sections={n_sec}  tables={n_tab}  figures={n_fig}  "
          f"equations={n_eq}")
    print(f"wrote {os.path.basename(DST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())