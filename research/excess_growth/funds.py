"""S3 (the nine SPDR sector funds), S4 (French's 30 industry portfolios, daily) and S5 (RSP - SPY), the .md's
"Survivorship".

    python3 -I -B funds.py old     # S3 2007..2025-06, S4 1927..2026-08 (downloads the zip once), S5 2006..2025-06
    python3 -I -B funds.py new     # S3 and S5 on the unseen months (called by unseen.py)

Writes funds_<tag>.txt / .json next to this script.
"""
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402
import run as RUN  # noqa: E402

np = C.np
FRENCH_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_daily_CSV.zip'
FRENCH_DIR = C.WORK / 'raw' / 'french30'
ERAS = (('1927-1962', 1927, 1962), ('1963-2007', 1963, 2007), ('2008-2019', 2008, 2019), ('2020-2026-08', 2020, 2026))


def lite(tag):
    E = C.c2eval.load_export(C.EXPORTS[tag], arrays=())
    return {'E': None, 'days': E['days'], 'iso': np.array([C.c2eval.iso(d) for d in E['days']]), 'div': False}


def store_funds(tag, D, tickers):
    S = C.c2eval.load_store(C.STORES[tag])
    by = {s['ticker']: s['id'] for s in S['syms'] if s['kind'] == 'etf'}
    ids = [by[t] for t in tickers]
    close = C.c2eval.store_panel(S, ids, D['days'], 'close').astype(np.float64)
    vol = C.c2eval.store_panel(S, ids, D['days'], 'volume').astype(np.float64)
    close[~(close > 0)] = np.nan
    fin = np.isfinite(close)
    T = close.shape[1]
    last = np.where(fin.any(1), T - 1 - np.argmax(fin[:, ::-1], 1), -1)
    return close, close * vol, last


def fund_periods(D, dec, labels, close, dv, last, last_end=None, rw=True):
    P = C.periods(D, dec, last_end=last_end)
    per = []
    for p, lab in zip(P, labels):
        cols = close[:, p['t0']:p['end'] + 1]
        rows = np.nonzero(np.isfinite(cols[:, 0]))[0]
        H = C.holding(D, rows, p, close=close, dv=dv, div=False, last=last)
        if p['t0'] >= C.WIN:
            X = C.window_logret(D, rows, p['t0'], close=close, div=False)
            ok = np.isfinite(X).sum(1) >= C.MIN_RET
            gam = C.gamma_star(X) if ok.all() else float('nan')
            Z = C.demean_fill(X)
        else:
            gam, Z = float('nan'), None
        r = {'label': lab, 'p': p, 'rows': rows, 'H': H, 'Z': Z, 'gamma': gam,
             'days': np.arange(p['s'] + 1, p['end'] + 1), 'ports': {}}
        for name in ('R1d', 'R1m', 'B1', 'R0d', 'R0m', 'B0'):
            kind, delay = RUN.PORTS[name]
            g, t, w0, we = C.run_period(H, kind, delay)
            r['ports'][name] = {'gross': g[0], 'turn': t[0], 'w0': w0[0], 'wend': we[0]}
        if rw and Z is not None and np.isfinite(gam):
            e = RUN.e1_period(r)
            r['e1'] = {k: e[k] for k in ('pred', 'EG_d', 'EG_m', 'J', 'gap_d0', 'gap_d1', 'gap_m1')}
            r['e1']['rw_gap_d'] = float(e['sim_gap_d'].mean())
            r['e1']['rw_tot'] = e['sim_gap_d']
        else:
            n = H['n']
            g = np.log1p(H['R'][:, 1:]).sum(1).mean()
            lv = {k: np.log1p(r['ports'][k]['gross']).sum() for k in ('R0d', 'R0m', 'B0', 'R1d', 'R1m', 'B1')}
            r['e1'] = {'pred': gam * n, 'EG_d': lv['R0d'] - g, 'EG_m': lv['R0m'] - g, 'J': lv['B0'] - g,
                       'gap_d0': lv['R0d'] - lv['B0'], 'gap_d1': lv['R1d'] - lv['B1'], 'gap_m1': lv['R1m'] - lv['B1']}
        per.append(r)
    return per


def series(per):
    S = {}
    for nm in ('R1d', 'R1m', 'B1', 'R0d', 'B0'):
        days, g, t, pid = RUN.chained(per, nm)
        S[nm] = (g, t)
    return S, days, pid


def span_tests(S, mask):
    out = {}
    for cname, a, b in (('daily', 'R1d', 'B1'), ('monthly', 'R1m', 'B1'), ('delay 0 daily', 'R0d', 'B0')):
        for cost, ctag in ((0.0, 'gross'), (C.COST, '5 bps')):
            d = (np.log1p(C.net(*S[a], cost)) - np.log1p(C.net(*S[b], cost)))[mask]
            out[f'{cname} {ctag}'] = C.test_series(d) if mask.sum() > 50 else {'per_year': 252 * float(d.mean()), 'days': int(mask.sum())}
    return out


def e1_sums(per, sel):
    keys = ('pred', 'EG_d', 'EG_m', 'J', 'gap_d0', 'gap_d1', 'gap_m1')
    out = {k: float(np.nansum([per[i]['e1'][k] for i in sel])) for k in keys}
    if all('rw_tot' in per[i]['e1'] for i in sel) and sel:
        tot = np.sum([per[i]['e1']['rw_tot'] for i in sel], 0)
        out['rw_mean'] = float(tot.mean())
        out['rw_p2_5'] = float(np.percentile(tot, 2.5))
        out['rw_p97_5'] = float(np.percentile(tot, 97.5))
        out['rw_pct'] = float((tot < out['gap_d0']).mean() * 100)
    return out


# ---- S4 --------------------------------------------------------------------------------------------------------
def french30():
    FRENCH_DIR.mkdir(parents=True, exist_ok=True)
    zp = FRENCH_DIR / '30_Industry_Portfolios_daily_CSV.zip'
    info = {'url': FRENCH_URL}
    if not zp.exists():
        req = urllib.request.Request(FRENCH_URL, headers={'User-Agent': 'Mozilla/5.0 (research; numpy)'})
        data = urllib.request.urlopen(req, timeout=120).read()
        zp.write_bytes(data)
        info['downloaded'] = C.time.strftime('%Y-%m-%d %H:%M:%S %Z')
        (FRENCH_DIR / 'downloaded.txt').write_text(info['downloaded'] + '\n')
    else:
        info['downloaded'] = (FRENCH_DIR / 'downloaded.txt').read_text().strip()
    raw = zp.read_bytes()
    info['sha256'] = hashlib.sha256(raw).hexdigest()
    info['bytes'] = len(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        info['members'] = names
        text = z.read(names[0]).decode('latin-1')
    lines = text.splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith(',') and 'Food' in l)
    head = [h.strip() for h in lines[k].split(',')[1:]]
    info['title_line'] = next((l.strip() for l in lines[max(0, k - 3):k] if l.strip()), '')
    dates, vals = [], []
    for l in lines[k + 1:]:
        if not l.strip():
            break
        f = l.split(',')
        dates.append(f[0].strip())
        vals.append([float(x) for x in f[1:]])
    V = np.array(vals).T                                       # [30, T] percent
    miss = (V <= -99.99 + 1e-9) | (V <= -999 + 1e-9)
    R = V / 100.0
    R[miss] = np.nan
    iso = np.array([f'{d[:4]}-{d[4:6]}-{d[6:]}' for d in dates])
    days = np.array([(np.datetime64(s) - np.datetime64('1970-01-01')).astype(int) for s in iso])
    info.update({'industries': head, 'first': iso[0], 'last': iso[-1], 'n_days': len(iso), 'n_missing': int(miss.sum())})
    # a total-return index as "closes"; a missing return leaves the close missing that day
    close = np.full((R.shape[0], R.shape[1] + 1), np.nan)
    close[:, 0] = 1.0
    lv = np.log1p(np.nan_to_num(R))
    close[:, 1:] = np.exp(np.cumsum(lv, 1))
    close[:, 1:][miss] = np.nan
    # prepend a pseudo day so the first return has a base
    days = np.concatenate([[days[0] - 1], days])
    iso = np.concatenate([['base'], iso])
    return close, days, iso, info


def s4():
    close, days, iso, info = french30()
    D = {'E': None, 'days': days, 'iso': iso, 'div': False}
    T = close.shape[1]
    years = sorted(set(int(s[:4]) for s in iso[1:]))
    dec = [max(i for i in range(1, T) if iso[i][:4] == str(y)) for y in years[:-1]]
    labels = [str(y + 1) for y in years[:-1]]
    dv = np.ones_like(close)
    last = np.full(close.shape[0], T - 1)
    P = C.periods(D, dec, last_end=T - 1)
    # an industry with a missing value in a period is left out of that period
    per = []
    for p, lab in zip(P, labels):
        ok = np.isfinite(close[:, p['t0']:p['end'] + 1]).all(1)
        rows = np.nonzero(ok)[0]
        cl = np.where(ok[:, None], close, np.nan)
        pp = fund_periods(D, [p['t0']], [lab], cl, dv, last, last_end=p['end'], rw=False)[0]
        pp['rows'] = rows
        per.append(pp)
    S, d, pid = series(per)
    lab_day = np.array([labels[k] for k in pid])
    out = {'info': info, 'eras': {}}
    for name, a, b in ERAS:
        m = np.isin(lab_day, [str(y) for y in range(a, b + 1)])
        sel = [k for k, l in enumerate(labels) if a <= int(l) <= b]
        out['eras'][name] = {'tests': span_tests(S, m), 'e1': e1_sums(per, sel), 'years': len(sel)}
    m = iso[d] >= '2025-07-01'
    out['eras']['2025-07..2026-08 (reported separately)'] = {'tests': span_tests(S, m), 'years': None}
    out['periods'] = [{'label': r['label'], 'K': len(r['rows']), **{k: float(v) for k, v in r['e1'].items() if k != 'rw_tot'}}
                      for r in per]
    return out


# ---- S5 --------------------------------------------------------------------------------------------------------
def s5(tag, D, close, labels_years):
    """RSP - SPY (store, both without dividends after about 2022-12) by calendar year, beside P1 without div_fix
    minus the store's SPY over the same days."""
    rsp, spy = close
    for x in (rsp, spy):
        first = int(np.argmax(np.isfinite(x)))
        for j in range(first + 1, len(x)):
            if not np.isfinite(x[j]):
                x[j] = x[j - 1]
    ser = np.load(C.WORK / f'series_{tag}.npz')
    sd, net1 = ser['days'], C.net(ser['R1d_nodiv_gross'], ser['R1d_nodiv_turn'], C.COST)
    gross1 = ser['R1d_nodiv_gross']
    iso = D['iso']
    rows = []
    for y in labels_years:
        idx = np.nonzero(np.array([s[:4] for s in iso]) == str(y))[0]
        if not len(idx):
            continue
        a = idx[0] - 1
        b = idx[-1]
        if tag == 'new' and y == 2025:
            a = int(np.nonzero(iso == '2025-06-30')[0][0])
        row = {'year': y, 'rsp': 100 * (rsp[b] / rsp[a] - 1), 'spy': 100 * (spy[b] / spy[a] - 1)}
        row['rsp_spy'] = row['rsp'] - row['spy']
        m = (sd >= idx[0]) & (sd <= b)
        if m.any():
            a2 = int(sd[m][0]) - 1
            row['p1'] = 100 * (np.prod(1 + net1[m]) - 1)
            row['p1_gross'] = 100 * (np.prod(1 + gross1[m]) - 1)
            row['spy_same'] = 100 * (spy[b] / spy[a2] - 1)
            row['rsp_same'] = 100 * (rsp[b] / rsp[a2] - 1)
            row['p1_spy'] = row['p1'] - row['spy_same']
            row['p1_rsp'] = row['p1'] - row['rsp_same']
        rows.append({k: (float(v) if not isinstance(v, int) else v) for k, v in row.items()})
    both = [r for r in rows if 'p1' in r]
    mid = [r for r in both if 2008 <= r['year'] <= 2022]
    return {'rows': rows, 'mean_rsp_spy_all': float(np.mean([r['rsp_spy'] for r in rows])),
            'mean_p1_rsp_2008_2022': float(np.mean([r['p1_rsp'] for r in mid])) if mid else None,
            'mean_rsp_spy_2008_2022': float(np.mean([r['rsp_same'] - r['spy_same'] for r in mid])) if mid else None,
            'mean_rsp_spy_p1years': float(np.mean([r['rsp_same'] - r['spy_same'] for r in both])) if both else None,
            'mean_p1_spy': float(np.mean([r['p1_spy'] for r in both])) if both else None,
            'mean_p1_rsp': float(np.mean([r['p1_rsp'] for r in both])) if both else None}


def main(tag):
    Dl = lite(tag)
    out = {'tag': tag}
    close, dv, last = store_funds(tag, Dl, C.SECTOR)
    if tag == 'old':
        dec = [C.year_end_index(Dl, y) for y in range(2006, 2025)]
        labels = [str(y) for y in range(2007, 2026)]
        spans = {'2007-2019': range(2007, 2020), '2020-2025-06': range(2020, 2026)}
    else:
        a = int(np.nonzero(Dl['iso'] == '2025-06-27')[0][0])
        dec, labels = [a, C.year_end_index(Dl, 2025)], ['2025H2', '2026']
        spans = {'unseen': None}
    per = fund_periods(Dl, dec, labels, close, dv, last)
    S, d, pid = series(per)
    lab_day = np.array([labels[k] for k in pid])
    out['S3'] = {'periods': [{'label': r['label'], 'K': len(r['rows']), 'gamma_yr': 252 * r['gamma'],
                              **{k: float(v) for k, v in r['e1'].items() if k != 'rw_tot'}} for r in per], 'spans': {}}
    for sp, yrs in spans.items():
        m = np.ones(len(d), bool) if yrs is None else np.isin(lab_day, [str(y) for y in yrs])
        sel = list(range(len(per))) if yrs is None else [k for k, l in enumerate(labels) if int(l) in yrs]
        out['S3']['spans'][sp] = {'tests': span_tests(S, m), 'e1': e1_sums(per, sel)}
    C.log('S3 done')
    if tag == 'old':
        out['S4'] = s4()
        C.log('S4 done')
    rs, _, _ = store_funds(tag, Dl, ('RSP', 'SPY'))
    out['S5'] = s5(tag, Dl, rs, range(2006, 2026) if tag == 'old' else (2025, 2026))
    report(out, tag)
    js = json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
    (C.HERE / f'funds_{tag}.json').write_text(json.dumps(js, indent=1))


def report(out, tag):
    L = []
    say = L.append

    def tests_block(tests):
        for k, t in tests.items():
            if 't_nw21' in t:
                say(f"    {k:22s} {100 * t['per_year']:+8.3f}%/yr  NW t {t['t_nw21']:+6.2f}  p {t['p']:.4f}  ({t['days']} days)")
            else:
                say(f"    {k:22s} {100 * t['per_year']:+8.3f}%/yr  ({t['days']} days)")

    say(f'S3: the nine SPDR sector funds ({tag} store), rebalanced equal weight minus buy-and-hold, reset each year')
    for sp, v in out['S3']['spans'].items():
        say(f'  {sp}:')
        tests_block(v['tests'])
        e = v['e1']
        say(f"    E1 sums (log points): pred {e['pred']:.4f}, EG daily {e['EG_d']:.4f}, J {e['J']:.4f}, gap delay 0 "
            f"{e['gap_d0']:+.4f}" + (f"; random walk {e['rw_mean']:+.4f} ({e['rw_p2_5']:+.4f}..{e['rw_p97_5']:+.4f}), "
                                     f"percentile {e['rw_pct']:.1f}" if 'rw_mean' in e else ''))
    say('  by period: ' + '; '.join(f"{r['label']} gap {r['gap_d1']:+.4f} (pred {r['pred']:.4f})" for r in out['S3']['periods']))
    if 'S4' in out:
        i = out['S4']['info']
        say(f"\nS4: French's 30 industry portfolios, daily ({i['first']}..{i['last']}, {i['n_days']} days, "
            f"{i['n_missing']} missing values; sha256 {i['sha256']})")
        for name, v in out['S4']['eras'].items():
            say(f'  {name}:')
            tests_block(v['tests'])
            if v.get('e1'):
                e = v['e1']
                say(f"    E1 sums: pred {e['pred']:.4f}, EG daily {e['EG_d']:.4f}, J {e['J']:.4f}, gap delay 0 {e['gap_d0']:+.4f}")
    s = out['S5']
    say('\nS5: RSP - SPY (store, both price-only after about 2022-12), beside P1 without div_fix minus the store SPY')
    for r in s['rows']:
        say(f"  {r['year']}: RSP {r['rsp']:+7.2f} SPY {r['spy']:+7.2f} RSP-SPY {r['rsp_spy']:+6.2f}" +
            (f" | same days: P1 {r['p1']:+7.2f} RSP {r['rsp_same']:+7.2f} SPY {r['spy_same']:+7.2f} P1-SPY "
             f"{r['p1_spy']:+6.2f} P1-RSP {r['p1_rsp']:+6.2f}" if 'p1' in r else ''))
    say(f"  means: RSP-SPY all years {s['mean_rsp_spy_all']:+.2f}; over P1's years RSP-SPY {s['mean_rsp_spy_p1years']:+.2f}, "
        f"P1-SPY {s['mean_p1_spy']:+.2f}, P1-RSP {s['mean_p1_rsp']:+.2f}"
        + (f"; 2008-2022 only: P1-RSP {s['mean_p1_rsp_2008_2022']:+.2f}, RSP-SPY {s['mean_rsp_spy_2008_2022']:+.2f}"
           if s.get('mean_p1_rsp_2008_2022') is not None else ''))
    say('  Dividends in the store (checked against Yahoo for SPY, TLT, LQD, IWM): funds are price-only before 2011 and '
        'adjusted from 2011; in 2023-2025 SPY is adjusted, LQD and IWM are not. RSP has no Yahoo series here.')
    txt = '\n'.join(L) + '\n'
    (C.HERE / f'funds_{tag}.txt').write_text(txt)
    print(txt, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
