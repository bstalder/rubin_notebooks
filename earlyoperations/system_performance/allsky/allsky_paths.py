"""Filesystem locations shared by the all-sky analysis scripts."""

import os
from pathlib import Path


ALLSKY_OUT = Path(
    os.environ.get(
        "ALLSKY_OUT",
        Path(__file__).resolve().parents[3] / "data" / "allsky",
    )
)
