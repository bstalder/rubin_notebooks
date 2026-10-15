"""Rubin GEO satellite brightness -> physical flux units -> equivalent source power.

The chain, with every step invertible so a lab number can be pushed either direction:

    AB mag  <->  f_nu [nJy, W/m^2/Hz]  <->  in-band irradiance F [W/m^2]
            <->  photon irradiance [ph/s/m^2]  <->  Rubin aperture rate [ph/s]
            <->  equivalent emitted power at GEO range P [W]  (isotropic or beamed)

Only the last step needs an assumption about how the source radiates; everything
before it is exact given the bandpass. That is deliberate: an unbuilt calibration
source has no defined beam yet, so results are reported as irradiance (assumption-free)
AND as required power vs beam width (a curve, not a single number).
"""

import numpy as np

# Rubin/LSST bandpasses: effective wavelength and effective width [nm].
# (lam_eff and the equivalent-rectangular widths of the total system throughput.)
BANDS = {
    "u": dict(lam=368.6, width=47.3),
    "g": dict(lam=480.3, width=141.1),
    "r": dict(lam=622.2, width=124.6),
    "i": dict(lam=754.5, width=100.3),
    "z": dict(lam=868.7, width=84.6),
    "y": dict(lam=971.0, width=61.8),
}
H = 6.62607015e-34  # J s
C = 2.99792458e8  # m/s
JY = 1e-26  # W m^-2 Hz^-1
AB_ZP_JY = 3631.0  # AB zero point

RUBIN_D_EFF = 6.423  # m, effective (obscured) aperture
RUBIN_AREA = np.pi * (RUBIN_D_EFF / 2) ** 2  # m^2  ~ 32.4
GEO_RANGE_M = 3.786e7  # m, slant range at ~50 deg elev (our hits)


def ab_to_fnu(ab):
    """AB mag -> f_nu [W m^-2 Hz^-1]."""
    return AB_ZP_JY * JY * 10 ** (-0.4 * np.asarray(ab, float))


def fnu_to_ab(fnu):
    """Inverse: f_nu [W m^-2 Hz^-1] -> AB mag."""
    return -2.5 * np.log10(np.asarray(fnu, float) / (AB_ZP_JY * JY))


def quantities(ab, band):
    """Full physical flux description of magnitude `ab` in `band` (top of atmosphere)."""
    b = BANDS[band]
    lam = b["lam"] * 1e-9
    dlam = b["width"] * 1e-9
    fnu = ab_to_fnu(ab)
    flam = fnu * C / lam**2  # W m^-2 m^-1
    F = flam * dlam  # W m^-2 in band
    Eph = H * C / lam  # J per photon
    return dict(
        band=band,
        AB=ab,
        lam_nm=b["lam"],
        width_nm=b["width"],
        f_nu_nJy=fnu / JY * 1e9,
        f_nu_W_m2_Hz=fnu,
        f_lam_W_m2_nm=flam * 1e-9,
        F_W_m2=F,
        phot_s_m2=F / Eph,
        rubin_phot_s=F / Eph * RUBIN_AREA,
        rubin_W=F * RUBIN_AREA,
    )


def required_power(ab, band, beam_fwhm_deg=None, range_m=GEO_RANGE_M):
    """Source power at GEO needed to deliver magnitude `ab` at the ground.

    isotropic: P = 4 pi r^2 F   (relevant for sunlight-scattering debris, diffuse)
    beamed:    P = Omega r^2 F  (relevant for a laser calibrator; Omega from FWHM cone)
    NOTE: this is in-band radiated power. A real source also needs its spectrum
    specified to relate in-band to total power; for a narrowband laser inside the
    band they coincide.
    """
    F = quantities(ab, band)["F_W_m2"]
    out = dict(
        F_W_m2=F, P_isotropic_W=F * 4 * np.pi * range_m**2, range_km=range_m / 1e3
    )
    if beam_fwhm_deg is not None:
        th = np.radians(beam_fwhm_deg)
        omega = 2 * np.pi * (1 - np.cos(th / 2))
        out.update(
            beam_fwhm_deg=beam_fwhm_deg,
            beam_sr=omega,
            P_beamed_W=F * omega * range_m**2,
            footprint_km=2 * range_m * np.tan(th / 2) / 1e3,
            gain_over_isotropic=4 * np.pi / omega,
        )
    return out


def mag_from_power(P_W, band, beam_fwhm_deg=None, range_m=GEO_RANGE_M):
    """INVERSE of required_power: lab-measured in-band power -> apparent AB mag at Rubin.
    This is the function to use once Landolt's lab radiometry exists."""
    if beam_fwhm_deg is None:
        F = P_W / (4 * np.pi * range_m**2)
    else:
        th = np.radians(beam_fwhm_deg)
        omega = 2 * np.pi * (1 - np.cos(th / 2))
        F = P_W / (omega * range_m**2)
    b = BANDS[band]
    lam = b["lam"] * 1e-9
    dlam = b["width"] * 1e-9
    flam = F / dlam
    fnu = flam * lam**2 / C
    return fnu_to_ab(fnu)


if __name__ == "__main__":
    import pandas as pd

    pd.set_option("display.width", 260)

    LADDER = [
        ("STARONE D2", "i", 10.49, "brightest confirmed comsat (station-kept)"),
        ("GOES 19", "i", 11.67, "pipeline validation anchor"),
        ("DIRECTV 15", "i", 11.94, "confirmed comsat"),
        ("SIRIUS FM-6", "g", 12.84, "confirmed comsat"),
        ("GALAXY 19", "u", 16.05, "best repeat-detected AB~16 analog"),
        ("INMARSAT 4-F3", "u", 17.88, "faintest single confirmed detection"),
        ("ASCENT per-exp UL", "y", 19.70, "12U CubeSat, 3sig UL single exposure"),
        ("ASCENT stack UL", "i", 21.60, "3sig UL, all July passes coadded"),
    ]
    rows = []
    for nm, bd, ab, note in LADDER:
        q = quantities(ab, bd)
        p = required_power(ab, bd)
        rows.append(
            dict(
                object=nm,
                band=bd,
                AB=ab,
                nJy=q["f_nu_nJy"],
                F_W_m2=q["F_W_m2"],
                ph_s_m2=q["phot_s_m2"],
                rubin_ph_s=q["rubin_phot_s"],
                P_iso_W=p["P_isotropic_W"],
                note=note,
            )
        )
    lad = pd.DataFrame(rows)
    print("=== 1. Measured Rubin GEO ladder in physical flux units ===")
    print(lad.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print(
        "\n=== 2. Same flux, as required SOURCE POWER vs beam width (band i, AB 17) ==="
    )
    print("    (a narrow beam needs far less power for the same ground irradiance)")
    br = []
    for fw in [None, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1]:
        p = required_power(17.0, "i", beam_fwhm_deg=fw)
        br.append(
            dict(
                beam_FWHM_deg=("isotropic" if fw is None else fw),
                footprint_km=p.get("footprint_km", np.nan),
                solid_angle_sr=p.get("beam_sr", 4 * np.pi),
                P_required_W=p.get("P_beamed_W", p["P_isotropic_W"]),
                gain=p.get("gain_over_isotropic", 1.0),
            )
        )
    print(pd.DataFrame(br).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print(
        "\n=== 3. INVERSE (the lab bridge): in-band source power -> AB mag seen by Rubin ==="
    )
    print(
        "    Use this once Landolt lab radiometry exists. Rows = candidate lab powers."
    )
    inv = []
    for P in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
        r = dict(P_in_band_W=P)
        for fw, lbl in [
            (None, "isotropic"),
            (10.0, "10deg"),
            (1.0, "1deg"),
            (0.1, "0.1deg"),
        ]:
            r[f"AB_{lbl}"] = mag_from_power(P, "i", beam_fwhm_deg=fw)
        inv.append(r)
    print(pd.DataFrame(inv).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== 4. Rubin single-exposure reference points ===")
    for ab in [16, 17, 18, 19, 20, 21, 22]:
        q = quantities(ab, "i")
        print(
            f"  AB {ab:2d} (i): {q['f_nu_nJy']:10.4g} nJy   {q['F_W_m2']:9.3e} W/m^2   "
            f"{q['phot_s_m2']:9.3e} ph/s/m^2   {q['rubin_phot_s']:9.3e} ph/s into Rubin   "
            f"{q['rubin_phot_s']*30:9.3e} ph in 30s"
        )
