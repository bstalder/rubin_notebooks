"""Camera-unit helpers shared by the all-sky scripts (unit labels come from build_unit_history.py / allsky_reduce.py)."""

import pandas as pd

from allsky_paths import ALLSKY_OUT

# Service visits that changed focus without a body/lens swap: a new stint starts on that night.
# 2026-10-16 (daytime, between 12:00 and 17:00 UTC): 8323/0552 refocused and zoom reset to the 8 mm end
# (star FWHM 2.6 -> 1.4 px, plate scale 848 -> 752 px/rad).
BREAKS = {"8323/0552": (20261016,)}
# Clear-sky reference for a stint too young for its own table (cloud_metric.table_for): a stint with the same
# focus and plate scale. 9118/0165 in 2022-02..05: FWHM 1.4 px, f 750 px/rad, same lens model at 8 mm, f/4.
PROXY = {"8323/0552@20261016": "9118/0165@20220223"}


def optics(unit):
    """Body/lens part of a unit label ('8323/0552/8' -> '8323/0552'). EXIF FocalLength is the zoom reading rounded to
    1 mm and flips between 8 and 9 on the same mount (2026-10-12), so seeds and completeness tables ignore it.
    """
    return unit.rsplit("/", 1)[0] if unit and unit.count("/") == 2 else unit


def stint_keys():
    """{(day_obs, body/lens): '<body/lens>@<first day_obs>'} -- the contiguous service stint each night belongs to.

    The same body + lens came back after other units several times (8323/0552: 2022-05, 2023-11, 2026-10), each time
    re-mounted and refocused, so a stint is the unit of constant geometry and focus. A swap night is in two stints.
    BREAKS adds stints for refocus visits that kept the same body + lens.
    """
    path = ALLSKY_OUT / "index" / "unit_nights.parquet"
    if not path.exists():
        return {}
    out, cur, start = {}, None, None
    for r in pd.read_parquet(path).sort_values("day_obs").itertuples():
        seen = [
            f"{str(r.first_body)[-4:]}/{str(r.first_lens)[-4:]}",
            f"{str(r.last_body)[-4:]}/{str(r.last_lens)[-4:]}",
        ]
        for o in dict.fromkeys(seen):
            if o != cur or (
                int(r.day_obs) in BREAKS.get(o, ()) and start != int(r.day_obs)
            ):
                cur, start = o, int(r.day_obs)
            out[(int(r.day_obs), o)] = f"{o}@{start}"
    return out
