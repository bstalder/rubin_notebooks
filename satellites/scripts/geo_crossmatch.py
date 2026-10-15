import numpy as np, pandas as pd
from skyfield.api import EarthSatellite, load, wgs84

ts = load.timescale()
rubin = wgs84.latlon(-30.244633, -70.749417, 2663.0)
FOV = 1.75

th = pd.read_parquet("/tmp/geo_tle_history.parquet")
exp = (
    pd.read_parquet("/tmp/dp2_window_exposures.parquet")
    .dropna(subset=["s_ra", "s_dec"])
    .copy()
)
exp["t"] = pd.to_datetime(exp.exp_midpt, utc=True)
exp = exp.sort_values("t").reset_index(drop=True)
exp["ra_r"] = np.radians(exp.s_ra)
exp["dec_r"] = np.radians(exp.s_dec)

# per-night exposure blocks + precomputed skyfield Time per night
nights = {}
for dob, gi in exp.groupby("day_obs"):
    nights[dob] = (
        gi.index.values,
        ts.from_datetimes(gi.t.dt.to_pydatetime().tolist()),
        gi.ra_r.values,
        gi.dec_r.values,
        gi.t.iloc[len(gi) // 2],
    )
print(
    f"{len(nights)} nights, {len(exp)} exposures, {th.NORAD_CAT_ID.nunique()} objects",
    flush=True,
)

rows = []
for nid, g in th.groupby("NORAD_CAT_ID"):
    g = g.sort_values("epoch").reset_index(drop=True)
    ep = g.epoch.values
    name = g.OBJECT_NAME.iloc[0]
    mm = float(g.TLE_LINE2.iloc[len(g) // 2][52:63])
    drift = (mm - 1.00273790) * 360.0
    tot = 0
    for dob, (idxs, T, rae, dece, tmid) in nights.items():
        j = int(np.abs(ep - np.datetime64(tmid.tz_localize(None))).argmin())
        age = abs(
            (np.datetime64(tmid.tz_localize(None)) - ep[j]) / np.timedelta64(1, "D")
        )
        if age > 3:  # no fresh element set for this night
            continue
        try:
            s = EarthSatellite(g.TLE_LINE1[j], g.TLE_LINE2[j], name, ts)
            d = (s - rubin).at(T)
            alt = d.altaz()[0].degrees
            rd = d.radec()
            ras, decs = rd[0].radians, rd[1].radians
        except Exception:
            continue
        cosd = np.sin(decs) * np.sin(dece) + np.cos(decs) * np.cos(dece) * np.cos(
            ras - rae
        )
        sep = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
        hit = (sep < FOV) & (alt > 20)
        if hit.any():
            sub = exp.loc[idxs[hit]].copy()
            sub["sep_deg"] = sep[hit]
            sub["sat_alt"] = alt[hit]
            sub["norad"] = int(nid)
            sub["sat_name"] = name
            sub["tle_age_days"] = age
            sub["drift_deg_day"] = drift
            rows.append(sub)
            tot += int(hit.sum())
    if tot:
        print(
            f"{name[:28]:28s} ({nid}): {tot:4d} hits  drift {drift:+7.2f} deg/d",
            flush=True,
        )

out = pd.concat(rows, ignore_index=True)
out.to_parquet("/tmp/geo_hits_hist.parquet")
print(f"\nTOTAL {len(out)} hits / {out.norad.nunique()} objects")
