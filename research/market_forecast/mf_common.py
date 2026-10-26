"""Market forecast (experiments/untested-luca-market-forecast.md): shared settings, loaders, inputs and statistics.

The rules are fixed in the plan ("Rules"); the choices it leaves open are in its "Decisions" section. Months are
integers (year * 12 + month - 1). A forecast at origin month L is made when BEA publishes month L; it forecasts
y_h(L) = log sum X(L+1..L+h) - log sum X(L+1-12..L+h-12) from g_h(L) = log sum X(L-h+1..L) - log sum X(L-h-11..L-12)
and the inputs' month-L values.

This folder is held outside the repo (correlation-pending/). It reuses the earlier alcohol test's loaders
(../alcohol_demand/ad_common.py: FRED, Google Trends, Yahoo) and its news series, read-only.

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation-pending/research/market_forecast')
    import mf_common as M
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'alcohol_demand'))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402

WORK = Path.home() / '.local/share/correlation2/experiments/market_forecast'
RAW = WORK / 'raw'

# ---- Fixed by the plan --------------------------------------------------------------------------------------------
MARKETS = {'W': 'DWINRA', 'B': 'DMLTRA', 'S': 'DLIQRA', 'R': 'DAPMRA'}          # quantity indexes
DOLLARS = {'W': 'DWINRC', 'B': 'DMLTRC', 'S': 'DLIQRC', 'R': 'DAPMRC'}
NAMES = {'W': 'wine', 'B': 'beer', 'S': 'spirits', 'R': 'restaurants and bars'}
PRICE = {'W': 'CUUR0000SEFW03', 'B': 'CUUR0000SEFW01', 'S': 'CUUR0000SEFW02', 'R': 'CUUR0000SEFX'}
INPUTS = ('E1', 'E2', 'E3', 'P', 'T1', 'T2', 'T3', 'LG', 'A')
THESES = {'economy': ('E1', 'E2', 'E3'), 'prices': ('P',), 'social trends': ('T1', 'T2'),
          'new markets': ('T3', 'LG'), 'age': ('A',)}
MIN_FIT = 36
NW_LAGS = 12
BLOCK = 24
N_BOOT = 10_000
N_NOISE = 200
SEED = 20261026
DEV = ('2008-01', '2019-12')            # by the target's last month, L + h
CHECK = ('2020-01', '2025-06')
UNSEEN = ('2025-07', '2026-08')
# Recreational cannabis decisions (the earlier test's C5 list); South Dakota's 2020 measure was struck down.
LEGAL = {'2012-11-06': ('CO', 'WA'), '2014-11-04': ('AK', 'OR', 'DC'), '2016-11-08': ('CA', 'NV', 'MA', 'ME'),
         '2018-01-22': ('VT',), '2018-11-06': ('MI',), '2019-06-25': ('IL',), '2020-11-03': ('AZ', 'MT', 'NJ'),
         '2021-03-31': ('NY',), '2021-04-07': ('VA',), '2021-04-12': ('NM',), '2021-06-22': ('CT',),
         '2022-05-25': ('RI',), '2022-11-08': ('MD', 'MO'), '2023-04-24': ('DE',), '2023-05-30': ('MN',),
         '2023-11-07': ('OH',)}
STATE_FIPS = {'AK': 2, 'AZ': 4, 'CA': 6, 'CO': 8, 'CT': 9, 'DE': 10, 'DC': 11, 'IL': 17, 'ME': 23, 'MD': 24, 'MA': 25,
              'MI': 26, 'MN': 27, 'MO': 29, 'MT': 30, 'NV': 32, 'NJ': 34, 'NM': 35, 'NY': 36, 'OH': 39, 'OR': 41,
              'RI': 44, 'VT': 50, 'VA': 51, 'WA': 53}
CE_GROUPS = ((2, 21, 24), (3, 25, 34), (4, 35, 44), (5, 45, 54), (6, 55, 64), (8, 65, 74), (9, 75, 85))

mkey, mindex = K.mkey, K.mindex


# ---- Loaders: each returns a dict 'YYYY-MM' -> value --------------------------------------------------------------

def bea(code):
    out = {}
    with open(RAW / 'NipaDataM.txt') as fh:
        for r in csv.reader(fh):
            if r and r[0] == code:
                out[f'{r[1][:4]}-{r[1][5:7]}'] = float(r[2].replace(',', ''))
    assert out, code
    return out


_BLS = None


def bls(series):
    global _BLS
    if _BLS is None:
        _BLS = json.loads((RAW / 'bls_api.json').read_text())
    out = {}
    for r in _BLS:
        if r['series'] == series and r['value'] not in ('-', ''):
            key = r['year'] if r['period'] == 'A01' else f"{r['year']}-{r['period'][1:]}"
            out[key] = float(r['value'])
    assert out, series
    return out


def fred_raw(name):
    out = {}
    for r in list(csv.reader(open(RAW / f'{name}.csv')))[1:]:
        if r[1] not in ('', '.'):
            out[r[0][:7]] = float(r[1])
    return out


TTB = {'wine': ('Wine_monthly_data_csv.csv', '2-Withdrawals', '0-Category Total', '2-Taxable Withdrawals'),
       'beer': ('Beer_National_Report_Monthly_csv.csv', '2-Removals', '0-Category Total', '2-Taxable*'),
       'spirits': ('Distilled_Spirits_monthly_data_csv.csv', '2-Withdrawals', '0-Category Total', '2-Taxable')}


def ttb(kind):
    f, g, c, d = TTB[kind]
    out = {}
    for r in csv.DictReader(open(RAW / f, encoding='utf-8-sig')):
        if (r['Statistical_Group'], r['Statistical_Category'], r['Statistical_Detail']) == (g, c, d):
            k = f"{int(r['Year']):04d}-{int(r['CY_Month_Number']):02d}"
            assert k not in out, (kind, k)
            if r['Stat_Redaction'] in ('TRUE', 'True') or r['Value'] == '':
                REDACTED.setdefault(kind, []).append(k)      # withheld by TTB: missing (Decisions 5)
                continue
            out[k] = float(r['Value'])
    return out


REDACTED = {}


def ma_sales():
    """Massachusetts adult-use retail sales in dollars, full months 2018-12..2026-09."""
    out = {}
    for r in csv.DictReader(open(RAW / 'ma_sales_au_gross.csv', encoding='utf-8-sig')):
        if r['DATATYPE'] == 'Retail':
            out[r['SALEDATE'][:7]] = out.get(r['SALEDATE'][:7], 0.0) + float(r['TOTAL_$'])
    return {k: v for k, v in out.items() if '2018-12' <= k <= '2026-09'}


def ma_price():
    return {r['YEARMONTH']: float(r['AVERAGERETAILPRICEPERGM'])
            for r in csv.DictReader(open(RAW / 'ma_sales_au_price_per_gram.csv', encoding='utf-8-sig'))}


# ---- Targets ------------------------------------------------------------------------------------------------------

def month_range(lo='2000-01', hi='2026-12'):
    return np.arange(mindex(lo), mindex(hi) + 1)


def sums(X, months, offsets):
    """sum of X over months L + j for j in offsets (nan if any is missing)."""
    out = np.full(len(months), np.nan)
    for i, L in enumerate(months):
        v = [X.get(mkey(L + j)) for j in offsets]
        if all(x is not None for x in v):
            out[i] = sum(v)
    return out


def target_and_history(X, months, h):
    y = np.log(sums(X, months, range(1, h + 1))) - np.log(sums(X, months, range(1 - 12, h + 1 - 12)))
    g = np.log(sums(X, months, range(-h + 1, 1))) - np.log(sums(X, months, range(-h + 1 - 12, 1 - 12)))
    return y, g


# ---- Inputs -------------------------------------------------------------------------------------------------------

def yoy(d, log=True):
    return K.yoy(d, log=log)


def spy_monthly():
    Y = K.yahoo_cache()
    d, p = Y['SPY']
    days = K.iso_days(d)
    out = {}
    for s, v in zip(days, p):
        out[s[:7]] = v            # the month's last trading day wins
    return out


def legal_share():
    """month -> share of the US population (50 states and DC, 2020 estimates) in states that had legalized by the
    month's end."""
    pop, total = {}, 0.0
    for r in csv.DictReader(open(RAW / 'NST-EST2025-ALLDATA.csv', encoding='latin-1')):
        if r['SUMLEV'] == '040' and int(r['STATE']) <= 56:
            pop[int(r['STATE'])] = float(r['POPESTIMATE2020'])
    total = sum(pop.values())
    assert len(pop) == 51, len(pop)
    out = {}
    for m in month_range('1990-01', '2026-12'):
        end = mkey(m) + '-31'
        states = {s for d, ss in LEGAL.items() if d <= end for s in ss}
        out[mkey(m)] = sum(pop[STATE_FIPS[s]] for s in states) / total
    return out


def population_by_age():
    """year -> array of 86 values (ages 0..84 and 85+), July estimates, from today's files."""
    out = {}
    for r in csv.DictReader(open(RAW / 'us-est00int-alldata.csv', encoding='latin-1')):
        if r['MONTH'] == '7' and 2000 <= int(r['YEAR']) <= 2009 and r['AGE'] != '999':
            out.setdefault(int(r['YEAR']), np.zeros(86))[min(int(r['AGE']), 85)] += float(r['TOT_POP'])
    for f, years in (('nc-est2020-agesex-res.csv', range(2010, 2020)), ('nc-est2025-agesex-res.csv', range(2020, 2026))):
        for r in csv.DictReader(open(RAW / f, encoding='latin-1')):
            if r['SEX'] == '0' and r['AGE'] != '999':
                for y in years:
                    out.setdefault(y, np.zeros(86))[min(int(r['AGE']), 85)] += float(r[f'POPESTIMATE{y}'])
    return out


def age_weights_by_year():
    """CE year -> weights w(a) for ages 0..85: spending per household of a's group / all households (ages < 21: 0)."""
    allh = bls('CXUALCBEVGLB0401M')
    grp = {g: bls(f'CXUALCBEVGLB04{g:02d}M') for g, _, _ in CE_GROUPS}
    out = {}
    for y in sorted(allh):
        w = np.zeros(86)
        for g, lo, hi in CE_GROUPS:
            w[lo:hi + 1] = grp[g][y] / allh[y]
        out[int(y)] = w
    return out


def age_shift(p):
    q = np.empty_like(p)
    q[0] = p[0]
    q[1:85] = p[0:84]
    q[85] = p[85] + p[84]
    return q


def age_input():
    """month L -> A(L): projected growth of the age-weighted population from year(L) to year(L) + 1."""
    pop, W = population_by_age(), age_weights_by_year()
    out = {}
    for L in month_range('2001-01', '2026-12'):
        yr, mo = L // 12, L % 12 + 1
        est = yr - 1 if mo >= 7 else yr - 2
        ce = yr - 1 if mo >= 9 else yr - 2
        if est not in pop or not any(y <= ce for y in W):
            continue
        w = np.mean([W[y] for y in W if y <= ce], axis=0)
        p = pop[est]
        for _ in range(yr - est):
            p = age_shift(p)
        out[mkey(L)] = float(np.log(age_shift(p) @ w) - np.log(p @ w))
    return out


def news_inputs():
    """The earlier test's N1 (share of alcohol headlines) and N2 (tone), standardized per source by its previous 36
    months and averaged over sources: month -> value."""
    sys.path.insert(0, str(HERE.parent / 'alcohol_demand'))
    import news as old_news
    src = json.loads((K.WORK / 'news_monthly.json').read_text())
    for s in src.values():
        for v in s.values():
            v['share'] = v['n_alc'] / v['n'] if v['n'] else None
    out = {}
    for k, key, need in (('N1', 'share', 36), ('N2', 'vader', 24)):
        per = [old_news.zscores(s, key, need) for s in src.values()]
        months = sorted({m for p in per for m in p})
        out[k] = {m: float(np.mean([p[m] for p in per if m in p])) for m in months}
    return out


def common_inputs():
    """The inputs that don't depend on the market: month -> value dicts."""
    U, S, C = K.read_fred('UNRATENSA'), K.read_fred('UMCSENT'), K.read_fred('CPIAUCNS')
    X = {'E1': yoy(U, log=False), 'E2': yoy(S), 'E3': yoy(spy_monthly())}
    for k, q in K.TREND_QUERY.items():
        X[k] = K.yoy(K.read_trends(q), floor=1.0)
    X['LG'] = yoy(legal_share(), log=False)
    X['A'] = age_input()
    X['_cpi'] = yoy(C)
    return X


def price_input(series_dict, X):
    yb = yoy(series_dict)
    return {k: yb[k] - X['_cpi'][k] for k in yb if k in X['_cpi']}


def at(d, months, lag=0, max_back=3):
    """Values at months L - lag; a month never published takes the latest earlier month with a value."""
    out = np.full(len(months), np.nan)
    for i, L in enumerate(months):
        for b in range(max_back + 1):
            v = d.get(mkey(L - lag - b))
            if v is not None:
                out[i] = v
                break
    return out


# ---- The model ----------------------------------------------------------------------------------------------------

def ols_forecast(Z, y, z0):
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return float(z0 @ b)


def oos(y, g, Xm, months, h, min_fit=MIN_FIT, sink=True):
    """Expanding out-of-sample forecasts at every origin with g and all inputs: f0 the prevailing mean, fb the
    benchmark (own history), fk[:, k] the single-input fits, fs the kitchen sink. Fits use origins s with
    s + h <= L where y, g and all inputs exist. g is the own-history column, or a matrix of them (the two-lag
    benchmark of Decisions 2: g_h(L) and g_h(L - 12)), used in the benchmark and in every fit."""
    G = g[:, None] if g.ndim == 1 else g
    n, nk = Xm.shape
    have = np.isfinite(G).all(1) & np.isfinite(Xm).all(1)
    sample = have & np.isfinite(y)
    f0, fb, fs = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    fk = np.full((n, nk), np.nan)
    for i in range(n):
        if not have[i]:
            continue
        el = np.nonzero(sample & (months + h <= months[i]))[0]
        if len(el) < min_fit:
            continue
        assert np.all(months[el] + h <= months[i])
        yy = y[el]
        f0[i] = yy.mean()
        one = np.ones(len(el))
        Zb, zb = np.column_stack([one, G[el]]), np.concatenate([[1.0], G[i]])
        fb[i] = ols_forecast(Zb, yy, zb)
        for k in range(nk):
            fk[i, k] = ols_forecast(np.column_stack([Zb, Xm[el, k]]), yy, np.append(zb, Xm[i, k]))
        if sink:
            fs[i] = ols_forecast(np.column_stack([Zb, Xm[el]]), yy, np.concatenate([zb, Xm[i]]))
    return {'f0': f0, 'fb': fb, 'fk': fk, 'fs': fs}


def combo(fk, cols):
    return fk[:, list(cols)].mean(1)


# ---- Statistics ---------------------------------------------------------------------------------------------------

def period_mask(months, h, lo, hi):
    end = months + h
    return (end >= mindex(lo)) & (end <= mindex(hi))


def evaluate(ys, fbs, fms, mask, boot=True, seed=SEED):
    """Pooled comparison over markets (lists of aligned arrays): the mean over markets of the Clark-West differential
    at each origin, its Newey-West t (NW_LAGS) and moving-block bootstrap p (BLOCK, N_BOOT); the pooled
    out-of-sample R^2 1 - sum e_m^2 / sum e_b^2 with a 95% block-bootstrap interval; MAE of both."""
    ok = mask.copy()
    for y, b, m in zip(ys, fbs, fms):
        ok &= np.isfinite(y) & np.isfinite(b) & np.isfinite(m)
    n = int(ok.sum())
    if n < 12:
        return {'n': n}
    E_b = np.array([(y - b)[ok] for y, b in zip(ys, fbs)])
    E_m = np.array([(y - m)[ok] for y, m in zip(ys, fms)])
    G = np.array([(b - m)[ok] for b, m in zip(fbs, fms)])
    d = (E_b ** 2 - E_m ** 2 + G ** 2).mean(0)
    se = K.nw_se(d, NW_LAGS)
    t = float(d.mean() / se) if se > 0 else float('nan')
    sb, sm = (E_b ** 2).sum(0), (E_m ** 2).sum(0)
    out = {'n': n, 'r2os': float(1 - sm.sum() / sb.sum()), 'cw_t': t, 'cw_p': K.p_upper(t),
           'mae_b': float(np.abs(E_b).mean()), 'mae_m': float(np.abs(E_m).mean())}
    if boot and n >= 2 * BLOCK:
        idx = K.block_index(n, BLOCK, N_BOOT, seed)
        dc = d - d.mean()
        out['cw_boot_p'] = float(np.mean(dc[idx].mean(1) >= d.mean()))
        r2 = 1 - sm[idx].sum(1) / sb[idx].sum(1)
        out['r2_lo95'], out['r2_hi95'] = float(np.percentile(r2, 2.5)), float(np.percentile(r2, 97.5))
    return out


def holm(ps):
    return K.holm(ps)


def dump(obj, path):
    K.dump(obj, path)


def fmt_ev(e):
    if e.get('n', 0) < 12:
        return f'n {e.get("n", 0)}: too few'
    s = f'n {e["n"]}, R2 {e["r2os"]:+.4f}, Clark-West t {e["cw_t"]:+.2f} (normal p {e["cw_p"]:.4f}'
    if 'cw_boot_p' in e:
        s += f', bootstrap p {e["cw_boot_p"]:.4f}; R2 95% {e["r2_lo95"]:+.3f}..{e["r2_hi95"]:+.3f}'
    return s + f'), MAE {100 * e["mae_b"]:.2f} -> {100 * e["mae_m"]:.2f} pts'


def panel(X, series, price, h, lag=0, extra=None, inputs=INPUTS):
    """Origins (months 2000-01..2026-08), target y_h, own history g_h and the input matrix (columns `inputs`, with P
    the market's own relative price), inputs taken lag months older (C7)."""
    months = month_range('2000-01', '2026-08')
    y, g = target_and_history(series, months, h)
    src = dict(X)
    src['P'] = price_input(price, X) if not isinstance(price, str) else X[price]
    if extra:
        src.update(extra)
    Xm = np.column_stack([at(src[k], months, lag) for k in inputs])
    return months, y, g, Xm


def two_lag(g):
    """[g_h(L), g_h(L - 12)] on the contiguous origin months (Decisions 2)."""
    g12 = np.full(len(g), np.nan)
    g12[12:] = g[:-12]
    return np.column_stack([g, g12])
