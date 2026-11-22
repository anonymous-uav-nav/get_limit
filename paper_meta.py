# -*- coding: utf-8 -*-
"""Paper metadata: title, venue, authors.  Supplied by the user, not inferred.

The author block is recorded here EXACTLY as given, because an earlier draft of
the outline guessed the author order from a pattern seen in a DIFFERENT paper in
this project (first author = one name, corresponding = the other).  That guess
was wrong for this paper: the user specified Yu Jian as FIRST author AND
corresponding, with Zheng Lijuan second and -- added later the same session --
as CO-CORRESPONDING.  A pattern observed elsewhere is not evidence about this
paper, and this module exists so the manuscript generator never has to guess.
"""
from __future__ import annotations

VERSION = "bcg-audit-paper-1.0"

# --------------------------------------------------------------------------
# title -- candidate 1 of the three in PAPER_OUTLINE.md, chosen for putting the
# contribution TYPE in the title so it is not read as yet another BCG method
# --------------------------------------------------------------------------
TITLE = ("Parameter Leverage on Emergent Collective Structure in a "
         "Bounded-Confidence Robot Swarm: A Pre-Registered Audit")

VENUE = "Drones"
VENUE_PUBLISHER = "MDPI"

# --------------------------------------------------------------------------
# authors -- verbatim from the user; do not reorder, do not infer roles
# --------------------------------------------------------------------------
AUTHORS = (
    {
        "name": "Jian Yu",
        "name_cn": "于健",
        "affiliation": "Kharkiv Polytechnic Institute",
        "affiliation_cn": "哈尔科夫理工学院",
        "orcid": "0009-0007-8990-8347",
        "emails": ("798733991@qq.com", "Jian.Yu@cs.khpi.edu.ua"),
        "roles": ("first", "corresponding"),
    },
    {
        "name": "Lijuan Zheng",
        "name_cn": "郑丽娟",
        "affiliation": "Sehan University",
        "affiliation_cn": "韩国世翰大学",
        "orcid": "0009-0002-2182-2313",
        "emails": ("supermassive@126.com",),
        "roles": ("second", "co-corresponding"),
    },
)

# --------------------------------------------------------------------------
# keywords, MDPI style
# --------------------------------------------------------------------------
KEYWORDS = (
    "bounded confidence",
    "robot swarm",
    "emergent collective structure",
    "parameter sensitivity",
    "pre-registered audit",
    "multi-agent systems",
)

# --------------------------------------------------------------------------
# the working name of the swarm model, used consistently in the manuscript
# --------------------------------------------------------------------------
MODEL_NAME = "bounded-confidence geometric swarm (BCGS)"

# --------------------------------------------------------------------------
# declarations -- placeholders that must be CONFIRMED by the author before
# submission, never auto-filled with a claim nobody checked
# --------------------------------------------------------------------------
DECLARATIONS_PENDING = (
    "funding          -- TODO author confirmation (no funding / grant number)",
    "acknowledgements -- TODO author confirmation (N/A or named contributions)",
    "conflicts        -- TODO author confirmation (declare none or list)",
    "CRediT           -- TODO author confirmation of role split",
    "data availability-- decide repository and DOI before submission",
    "AI disclosure    -- state tool and scope honestly, per MDPI policy",
)
