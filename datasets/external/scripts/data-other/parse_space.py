"""GFZ Kp/ap/Ap/SN/F10.7 and SILSO daily sunspot number -> normalized series."""
import datetime as dt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series

GFZ = os.path.join(RAW, "gfz", "Kp_ap_Ap_SN_F107_since_1932.txt")
GFZ_URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
SIL = os.path.join(RAW, "silso", "SN_d_tot_V2.0.csv")
SIL_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
TODAY_UTC = dt.datetime.now(dt.timezone.utc).date()

ap, kpmax, f107, sn_gfz, flag = {}, {}, {}, {}, {}
for line in open(GFZ):
    if line.startswith("#") or not line.strip():
        continue
    t = line.split()
    assert len(t) == 28, (len(t), line)
    d = dt.date(int(t[0]), int(t[1]), int(t[2]))
    if d >= TODAY_UTC:          # UT day not finished yet
        continue
    kps = [float(x) for x in t[7:15]]
    aps = [int(x) for x in t[15:23]]
    Ap, SN, Fobs, D = int(t[23]), int(t[24]), float(t[25]), int(t[27])
    flag[d] = D
    if all(k >= 0 for k in kps):
        kpmax[d] = max(kps)
    if Ap >= 0 and all(a >= 0 for a in aps):
        # Ap is the mean of the eight ap values (rounded); verify
        assert abs(Ap - sum(aps) / 8) <= 1.0, (d, Ap, aps)
        ap[d] = Ap
    if SN >= 0:
        sn_gfz[d] = SN
    if Fobs > 0:
        f107[d] = Fobs

prelim_from = min(d for d, D in flag.items() if D < 2)
common_notes = (f"GFZ file mixes definitive and preliminary values: Kp/Ap definitive through about "
                f"{min(d for d, D in flag.items() if D == 0) - dt.timedelta(days=1)}, nowcast after "
                f"(nowcast values are replaced by definitive ones, usually within a month; differences are small). "
                f"Values stored are the definitive ones a trader would only have seen as nowcasts. UT days.")

write_series("gfz_ap", sorted(ap.items()), name="Planetary geomagnetic Ap index (daily)",
             category="space_weather", subcategory="geomagnetic", region="global", source="GFZ Potsdam",
             url=GFZ_URL, frequency="daily", units="Ap index (unitless; x2 nT at 50 deg geomagnetic latitude)",
             transform="level", known_after_days=1, value_range=(0, 400),
             notes="Daily mean of the eight 3-hour ap values. Heavily right-skewed (storm days 30-400). "
                   "Krivelyova & Robotti (2003) found lower US returns in the days after geomagnetic storms. " + common_notes)
write_series("gfz_kp_max", sorted(kpmax.items()), name="Daily maximum of the 3-hour Kp index",
             category="space_weather", subcategory="geomagnetic", region="global", source="GFZ Potsdam",
             url=GFZ_URL, frequency="daily", units="Kp (0-9, thirds as .333/.667)", transform="level",
             known_after_days=1, value_range=(0, 9),
             notes="Max over the eight 3-hour Kp values of the UT day; Kp>=5 is a NOAA G1+ storm. " + common_notes)
storm = sorted((d, 1.0 if a >= 30 else 0.0) for d, a in ap.items())
write_series("gfz_storm_ap30", storm, name="Geomagnetic storm day indicator (Ap >= 30)",
             category="space_weather", subcategory="geomagnetic", region="global", source="GFZ Potsdam (derived)",
             url=GFZ_URL, frequency="daily", units="0/1", transform="level", known_after_days=1,
             value_range=(0, 1),
             notes="1 if daily Ap >= 30 (NOAA daily classification: minor storm or worse), else 0; the "
                   "storm-day definition used in the geomagnetic-storm/stock-return literature. Derived from gfz_ap. " + common_notes)
write_series("gfz_f107_obs", sorted(f107.items()), name="F10.7 solar radio flux, observed (daily)",
             category="space_weather", subcategory="solar", region="global",
             source="GFZ Potsdam (from DRAO / NRCan Penticton)", url=GFZ_URL, frequency="daily",
             units="solar flux units (1e-22 W m^-2 Hz^-1)", transform="logdiff", known_after_days=1,
             value_range=(50, 1000),
             notes="Local-noon (about 20 UT) observed flux at Penticton; follows the 11-year solar cycle. "
                   "Solar-activity series: a placebo, nothing in markets should follow it. "
                   "Missing days (instrument outages) are left out. Observed flux varies about +-3.4% "
                   "over the year with Earth-Sun distance (F10.7adj removes that). A few days exceed 500 sfu (e.g. 2001-12-28, 2003-11-04, 2005-09-09, 2006-12-06): solar radio bursts from major flares during the noon measurement, kept as published.")

# SILSO daily total sunspot number v2.0: year;month;day;decyear;SN;sd;nobs;definitive
sil, sil_prov = {}, []
for line in open(SIL):
    t = [x.strip() for x in line.split(";")]
    if len(t) < 8:
        continue
    d = dt.date(int(t[0]), int(t[1]), int(t[2]))
    v = int(t[4])
    if v < 0:
        continue
    sil[d] = v
    if t[7] == "0":
        sil_prov.append(d)
prov_from = min(sil_prov) if sil_prov else None
write_series("silso_sn_daily", sorted(sil.items()), name="International sunspot number, daily total (v2.0)",
             category="space_weather", subcategory="solar", region="global", source="WDC-SILSO, Royal Observatory of Belgium",
             url=SIL_URL, frequency="daily", units="sunspot number (v2.0 scale)", transform="diff",
             known_after_days=1, value_range=(0, 600),
             notes="PLACEBO: no plausible mechanism links sunspots to next-day US stock moves; a lead-lag hit here "
                   "measures the test's false-positive rate. Can be 0 at solar minimum, hence diff. "
                   f"Values from {prov_from} on are provisional (definitive flag 0). The whole series was "
                   "recalibrated (v2.0) in July 2015, so pre-2015 values differ from what was published then "
                   "(a scale change, harmless for correlation). Daily values before 1849 are sparse; missing (-1) left out. "
                   "SILSO updates this file monthly; the next-day estimate (EISN) is published daily.")

# cross-check SILSO vs GFZ's copy of SN
both = [d for d in sil if d in sn_gfz and d >= dt.date(2000, 1, 1)]
diff = [abs(sil[d] - sn_gfz[d]) for d in both]
mism = [(d, sil[d], sn_gfz[d]) for d in both if sil[d] != sn_gfz[d]]
print(f"\nSN cross-check 2000+: {len(both)} common days, {len(mism)} differ, max |diff| = {max(diff)}")
print("  sample mismatches:", mism[:8])
# storm frequencies
n = len(ap); s30 = sum(1 for a in ap.values() if a >= 30)
k5 = sum(1 for k in kpmax.values() if k >= 5)
print(f"storm days: Ap>=30 {s30}/{n} ({100*s30/n:.2f}%), Kpmax>=5 {k5}/{len(kpmax)} ({100*k5/len(kpmax):.2f}%)")
y2k = [d for d in ap if d >= dt.date(2000, 1, 1)]
print(f"  since 2000: Ap>=30 on {sum(1 for d in y2k if ap[d] >= 30)} of {len(y2k)} days")
