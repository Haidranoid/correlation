"""Computed astronomical series (no download): lunar phase and New York daylight.

Lunar: new-moon instants from Meeus, Astronomical Algorithms (2nd ed.) ch. 49 (accurate to
a few minutes); illuminated fraction from Meeus ch. 48 phase-angle formula (48.4). Both
evaluated at 16:00 America/New_York (the US close) each calendar day.
Daylight: NOAA sunrise equation (solar declination + -0.833 deg altitude for refraction and
the solar semi-diameter) at 40.7128 N, 74.0060 W.
"""
import datetime as dt, math, os, sys
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_series, daterange

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
START, END = dt.date(1990, 1, 1), dt.date(2026, 10, 18)
rad = math.radians

def jd_from_datetime(t):  # t aware datetime -> Julian Day (UT)
    t = t.astimezone(UTC)
    return 2440587.5 + t.timestamp() / 86400.0

def datetime_from_jd(jd):
    return dt.datetime.fromtimestamp((jd - 2440587.5) * 86400.0, UTC)

def delta_t_days(year):  # TT-UT, ~63.8 s (2000) to ~69.2 s (2025); linear is fine here
    return (63.8 + (year - 2000) * 0.22) / 86400.0

def new_moon_jde(k):
    """Meeus ch.49, new moon for integer k (k=0 near 2000-01-06). Returns JDE (TT)."""
    T = k / 1236.85
    jde = (2451550.09766 + 29.530588861 * k + 0.00015437 * T**2 - 0.000000150 * T**3
           + 0.00000000073 * T**4)
    E = 1 - 0.002516 * T - 0.0000074 * T**2
    M = rad(2.5534 + 29.10535670 * k - 0.0000014 * T**2 - 0.00000011 * T**3)
    Mp = rad(201.5643 + 385.81693528 * k + 0.0107582 * T**2 + 0.00001238 * T**3
             - 0.000000058 * T**4)
    F = rad(160.7108 + 390.67050284 * k - 0.0016118 * T**2 - 0.00000227 * T**3
            + 0.000000011 * T**4)
    Om = rad(124.7746 - 1.56375588 * k + 0.0020672 * T**2 + 0.00000215 * T**3)
    s = math.sin
    corr = (-0.40720 * s(Mp) + 0.17241 * E * s(M) + 0.01608 * s(2 * Mp) + 0.01039 * s(2 * F)
            + 0.00739 * E * s(Mp - M) - 0.00514 * E * s(Mp + M) + 0.00208 * E * E * s(2 * M)
            - 0.00111 * s(Mp - 2 * F) - 0.00057 * s(Mp + 2 * F) + 0.00056 * E * s(2 * Mp + M)
            - 0.00042 * s(3 * Mp) + 0.00042 * E * s(M + 2 * F) + 0.00038 * E * s(M - 2 * F)
            - 0.00024 * E * s(2 * Mp - M) - 0.00017 * s(Om) - 0.00007 * s(Mp + 2 * M)
            + 0.00004 * s(2 * Mp - 2 * F) + 0.00004 * s(3 * M) + 0.00003 * s(Mp + M - 2 * F)
            + 0.00003 * s(2 * Mp + 2 * F) - 0.00003 * s(Mp + M + 2 * F)
            + 0.00003 * s(Mp - M + 2 * F) - 0.00002 * s(Mp - M - 2 * F)
            - 0.00002 * s(3 * Mp + M) + 0.00002 * s(4 * Mp))
    A = [299.77 + 0.107408 * k - 0.009173 * T**2, 251.88 + 0.016321 * k, 251.83 + 26.651886 * k,
         349.42 + 36.412478 * k, 84.66 + 18.206239 * k, 141.74 + 53.303771 * k,
         207.14 + 2.453732 * k, 154.84 + 7.306860 * k, 34.52 + 27.261239 * k,
         207.19 + 0.121824 * k, 291.34 + 1.844379 * k, 161.72 + 24.198154 * k,
         239.56 + 25.513099 * k, 331.55 + 3.592518 * k]
    C = [0.000325, 0.000165, 0.000164, 0.000126, 0.000110, 0.000062, 0.000060, 0.000056,
         0.000047, 0.000042, 0.000040, 0.000037, 0.000035, 0.000023]
    add = sum(c * math.sin(rad(a)) for c, a in zip(C, A))
    return jde + corr + add

def new_moon_ut(k):
    jde = new_moon_jde(k)
    year = 2000 + k / 12.3685
    return jde - delta_t_days(year)

def illuminated_fraction(jd_ut):
    """Meeus 48.4 + 48.1: phase angle from mean elongation D, anomalies M, M'."""
    T = (jd_ut + delta_t_days(2000 + (jd_ut - 2451545) / 365.25) - 2451545.0) / 36525.0
    D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T**2 + T**3 / 545868 - T**4 / 113065000
    M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T**2 + T**3 / 24490000
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T**2 + T**3 / 69699 - T**4 / 14712000
    D, M, Mp = rad(D % 360), rad(M % 360), rad(Mp % 360)
    i = (180 - math.degrees(D) - 6.289 * math.sin(Mp) + 2.100 * math.sin(M)
         - 1.274 * math.sin(2 * D - Mp) - 0.658 * math.sin(2 * D) - 0.214 * math.sin(2 * Mp)
         - 0.110 * math.sin(D))
    return (1 + math.cos(rad(i))) / 2

# ---- validation against published new-moon times (UT) ----
checks = {  # USNO / NASA phase tables
    "2000-01-06 18:14": 0, "2024-01-11 11:57": None, "2025-06-25 10:31": None,
    "2008-10-28 23:14": None, "2017-08-21 18:30": None,
}
def k_near(date):
    return round((date - dt.date(2000, 1, 6)).days / 29.530588861)
print("new-moon validation (computed vs published, minutes):")
for s_ in checks:
    pub = dt.datetime.strptime(s_, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    k = k_near(pub.date())
    comp = datetime_from_jd(new_moon_ut(k))
    print(f"  {s_}  computed {comp:%Y-%m-%d %H:%M}  diff {(comp - pub).total_seconds() / 60:+.1f} min")
    assert abs((comp - pub).total_seconds()) < 15 * 60
# illuminated fraction sanity: ~0 at a new moon, ~1 at a full moon (2024-01-25 17:54 UT full)
nm = new_moon_ut(k_near(dt.date(2024, 1, 11)))
fm = jd_from_datetime(dt.datetime(2024, 1, 25, 17, 54, tzinfo=UTC))
print(f"  k at new moon 2024-01-11: {illuminated_fraction(nm):.4f}; at full moon 2024-01-25: "
      f"{illuminated_fraction(fm):.4f}")
assert illuminated_fraction(nm) < 0.002 and illuminated_fraction(fm) > 0.998
# Meeus example 48.a: 1992-04-12 0h TD -> k = 0.6786
ex = jd_from_datetime(dt.datetime(1992, 4, 12, 0, 0, tzinfo=UTC)) - delta_t_days(1992)
print(f"  Meeus ex.48.a k(1992-04-12) = {illuminated_fraction(ex):.4f} (book: 0.6786)")

# ---- daily series at 16:00 New York ----
k0 = k_near(START) - 2
nms = [new_moon_ut(k) for k in range(k0, k_near(END) + 3)]
illum, age = [], []
j = 0
for d in daterange(START, END):
    t = dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY)
    jd = jd_from_datetime(t)
    while nms[j + 1] <= jd:
        j += 1
    assert nms[j] <= jd < nms[j + 1]
    illum.append((d, round(illuminated_fraction(jd), 5)))
    age.append((d, round(jd - nms[j], 4)))

note_lunar = ("Computed, no download: Meeus (1998) Astronomical Algorithms ch. 48-49; new-moon "
              "instants validated against published times to within minutes. Evaluated at 16:00 "
              "New York time each calendar day (incl. weekends). Deterministic and known years in "
              "advance, hence known_after_days 0. Lunar-cycle 'moonstruck' effect: Yuan, Zheng & "
              "Zhu (2006) found lower returns around full moons. Near-placebo: a lead-lag hit at "
              "a non-29.5-day offset is noise.")
write_series("astro_moon_illum", illum, name="Moon illuminated fraction at the US close",
             category="astronomical", subcategory="lunar", region="global", source="computed",
             url="", frequency="daily", units="fraction 0-1 (0 new moon, 1 full moon)",
             transform="level", known_after_days=0, value_range=(0, 1), notes=note_lunar)
write_series("astro_moon_age", age, name="Days since the last new moon at the US close",
             category="astronomical", subcategory="lunar", region="global", source="computed",
             url="", frequency="daily", units="days (0 to about 29.8)", transform="level",
             known_after_days=0, value_range=(0, 30),
             notes="Sawtooth: resets at each new moon (a linear correlation treats day 29 and day 0 as far "
                   "apart). " + note_lunar)

# ---- NYC daylight hours (NOAA sunrise equation) ----
LAT = 40.7128
def daylight_hours(d):
    # fractional year (radians) at local solar noon
    doy = d.timetuple().tm_yday
    ndays = 366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365
    g = 2 * math.pi / ndays * (doy - 1 + (12 - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    phi = rad(LAT)
    cosH = (math.cos(rad(90.833)) / (math.cos(phi) * math.cos(decl))
            - math.tan(phi) * math.tan(decl))
    H = math.degrees(math.acos(cosH))
    return 2 * H / 15.0

day = [(d, round(daylight_hours(d), 4)) for d in daterange(START, END)]
# check against published NYC values: about 15h05m at June solstice, 9h15m at December solstice
jun = daylight_hours(dt.date(2024, 6, 20)); dec = daylight_hours(dt.date(2024, 12, 21))
print(f"NYC daylight 2024-06-20 {int(jun)}h{(jun % 1) * 60:.0f}m (published 15h05m); "
      f"2024-12-21 {int(dec)}h{(dec % 1) * 60:.0f}m (published 9h15m)")
assert abs(jun - (15 + 5 / 60)) < 3 / 60 and abs(dec - (9 + 15 / 60)) < 3 / 60
write_series("astro_daylight_nyc", day, name="Daylight hours in New York City",
             category="astronomical", subcategory="seasonal", region="US-NY",
             source="computed (NOAA sunrise equation)", url="", frequency="daily",
             units="hours (sunrise to sunset)", transform="level", known_after_days=0,
             value_range=(9, 15.5),
             notes="Computed, no download; 40.7128N 74.0060W, -0.833 deg solar altitude; checked against "
                   "published NYC solstice day lengths to within 3 minutes. Deterministic annual cycle, so "
                   "any lead-lag hit is a seasonality-in-returns finding, not a forecast. Kamstra, Kramer & "
                   "Levi (2003, AER) seasonal-affective-disorder effect; they also stress the change in "
                   "daylight around the equinoxes, so the test may want diff here.")
sad = []
for d, h in day:
    md = (d.month, d.day)
    fall_winter = md >= (9, 21) or md <= (3, 20)
    sad.append((d, round((24 - h) - 12, 4) if fall_winter else 0.0))
write_series("astro_sad_nyc", sad, name="Kamstra-Kramer-Levi SAD variable for New York",
             category="astronomical", subcategory="seasonal", region="US-NY",
             source="computed (from astro_daylight_nyc)", url="", frequency="daily",
             units="hours (night length minus 12, fall and winter only)", transform="level",
             known_after_days=0, value_range=(-1, 3),
             notes="KKL (2003) SAD_t = night hours - 12 for dates 21 Sep-20 Mar, 0 otherwise, at New York. "
                   "Deterministic seasonal series; see astro_daylight_nyc.")
