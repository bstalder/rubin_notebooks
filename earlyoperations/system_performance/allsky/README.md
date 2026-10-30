# allsky/ — moved

The all-sky camera reduction moved on 2026-10-30, with its git history, to
**[bstalder/rubin_allskycam](https://github.com/bstalder/rubin_allskycam)** for handover to DM.
That repository has the pipeline (`allsky_reduce.py`, the standard and deep cloud indexes, the
DREAM and nightly joins), its documentation and the list of open handover issues.

The scripts were last here at commit `607ccd1`. Products are unchanged and stay under `ALLSKY_OUT`
(USDF: `/sdf/data/rubin/user/stalder/allsky`).
