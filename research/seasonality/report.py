"""Tables for tests 1-3, the contrast, the periods and the published-claim comparison.

    python3 -I -B report.py old      -> results_old.txt   (book_old_raw, and book_old_fix once divfix.py has run)
    python3 -I -B report.py new      -> results_new.txt   (the unseen months, 2025-07..2026-10; once, at the end)

Alphas: monthly long-short returns regressed on French's FF5 (Mkt-RF, SMB, HML, RMW, CMA) + UMD, each factor
compounded over the exact holding window (the export days after e[i-1] through e[i]) from French's daily files.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, HERE, tstat, nw_t, ols, month_label, log  # noqa: E402

PUB = Path.home() / '.local/share/correlation2/experiments/published_data'
sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')

tag = sys.argv[1]
lines = []


def P(s=''):
    print(s, flush=True)
    lines.append(s)


def load(variant):
    p = WORK / f'book_{tag}_{variant}.npz'
    return dict(np.load(p)) if p.exists() else None


B = {'raw': load('raw'), 'fix': load('fix')}
z = B['raw']
months, e, days = z['months'], z['e'], z['days']

# factors per holding month (NaN where French's daily file ends)
FAC_NAMES = ('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'UMD')
FAC = None
if (PUB / 'READY').exists():
    import pubdata
    d5, f5 = pubdata.ff5_daily()
    du, fu = pubdata.umd_daily()
    daily = {k: pubdata.on_days(d5, f5[k], days) for k in FAC_NAMES[:5]}
    daily['UMD'] = pubdata.on_days(du, fu['UMD'], days)
    last_fac = int(pubdata.to_days(d5[-1:])[0])
    FAC = np.full((len(months), 6), np.nan)
    for i in range(1, len(months)):
        a, b = e[i - 1] + 1, e[i] + 1
        if days[b - 1] > last_fac:
            continue
        for c, k in enumerate(FAC_NAMES):
            x = daily[k][a:b]
            if np.isfinite(x).all():
                FAC[i, c] = np.prod(1 + x) - 1


def sel(bk, name, a, b, include_partial=True):
    """Positions in book `name` of holding months a..b (labels 'YYYY-MM')."""
    i = bk[f'{name}__i']
    mm = months[i]
    ya, ma = map(int, a.split('-'))
    yb, mb = map(int, b.split('-'))
    ok = (mm >= ya * 12 + ma - 1) & (mm <= yb * 12 + mb - 1)
    if not include_partial:
        ok &= ~bk[f'{name}__partial']
    return np.nonzero(ok)[0]


def stats(bk, name, idx, kind='net'):
    y = bk[f'{name}__{kind}'][idx]
    mean, t, n = tstat(y)
    out = dict(n=n, mean=100 * mean, t=t, nwt=nw_t(y), sharpe=mean / y.std(ddof=1) * np.sqrt(12) if n > 2 else np.nan,
               alpha=np.nan, ta=np.nan, nfac=0, L=100 * np.mean(bk[f'{name}__L'][idx] - bk['mkt'][bk[f'{name}__i'][idx]]),
               S=100 * np.mean(bk[f'{name}__S'][idx] - bk['mkt'][bk[f'{name}__i'][idx]]),
               turn=np.mean(bk[f'{name}__turn'][idx]), cost=100 * np.mean(bk[f'{name}__cost'][idx]))
    if FAC is not None:
        X = FAC[bk[f'{name}__i'][idx]]
        ok = np.isfinite(X).all(1)
        if ok.sum() >= 12:
            b, tb, nn, r2 = ols(y[ok], X[ok])
            out.update(alpha=100 * b[0], ta=tb[0], nfac=nn, betas=b[1:], tbetas=tb[1:])
    return out


HDR = (f"{'':44s} {'months':>6s} {'%/month':>8s} {'t':>6s} {'NW t':>6s} {'Sharpe':>6s} {'alpha%/m':>8s} {'t':>6s} "
       f"{'(n)':>5s} {'long-mkt':>8s} {'short-mkt':>9s}")


def row(label, s):
    P(f"{label:44s} {s['n']:6d} {s['mean']:8.3f} {s['t']:6.2f} {s['nwt']:6.2f} {s['sharpe']:6.2f} {s['alpha']:8.3f} "
      f"{s['ta']:6.2f} {s['nfac']:5d} {s['L']:8.3f} {s['S']:9.3f}")


def block(title, bk, name, periods, kinds=('net', 'gross')):
    P(f'-- {title}')
    P(HDR)
    res = {}
    for lab, a, b in periods:
        idx = sel(bk, name, a, b)
        if len(idx) < 3:
            continue
        for kind in kinds:
            s = stats(bk, name, idx, kind)
            res[(lab, kind)] = s
            row(f'{lab} {kind}', s)
    return res


def diff_stats(bk, a, b, kind='gross'):
    """same minus other_kln, month by month (both books exist)."""
    i1 = sel(bk, 'same', a, b)
    i2 = sel(bk, 'other_kln', a, b)
    m1 = bk['same__i'][i1]
    m2 = bk['other_kln__i'][i2]
    common = np.intersect1d(m1, m2)
    y = (bk[f'same__{kind}'][i1][np.isin(m1, common)] - bk[f'other_kln__{kind}'][i2][np.isin(m2, common)])
    mean, t, n = tstat(y)
    return n, 100 * mean, t


if tag == 'old':
    start = (HERE / 'start.txt').read_text().splitlines()[0]
    P(start)
    S0 = '2007-05'
    T1 = ('test 1: 2007-05..2019-12', S0, '2019-12')
    T1b = ('sensitivity: 2010-03..2019-12', '2010-03', '2019-12')
    T3 = ('test 3: 2020-01..2025-06', '2020-01', '2025-06')
    T3a = ('2020-01..2022-06', '2020-01', '2022-06')
    T3b = ('2022-07..2025-06', '2022-07', '2025-06')
    P(f"FF5 + UMD factors: {'loaded' if FAC is not None else 'NOT READY (alphas pending)'}")
    P('Holding month 2025-06 is partial in the old export (through 2025-06-27).')
    P()
    raw, fix = B['raw'], B['fix']
    P('== Test 1: the same-month book, universe, start month..2019-12 (net = after 5 bps per side) ==')
    r1 = block('same, raw closes', raw, 'same', [T1, T1b])
    s1 = r1[(T1[0], 'net')]
    pass1 = s1['t'] >= 2 and s1['alpha'] > 0
    P(f"PASS test 1 (net t >= 2 and FF5+UMD alpha > 0): {pass1}   (net t {s1['t']:.2f}, alpha {s1['alpha']:+.3f}%/month "
      f"t {s1['ta']:.2f})")
    if FAC is not None:
        P('   factor loadings (net, test 1): ' + ', '.join(f'{k} {b:+.2f} (t {t:+.2f})' for k, b, t in
                                                         zip(FAC_NAMES, s1['betas'], s1['tbetas'])))
    P()
    P('== Test 2: the 500 largest, same months ==')
    r2 = block('same500, raw closes', raw, 'same500', [T1, T1b])
    s2 = r2[(T1[0], 'net')]
    P(f"PASS test 2 (same sign as test 1, net): {np.sign(s2['mean']) == np.sign(s1['mean'])}  ({s2['mean']:+.3f}%/month)")
    P()
    P('== Test 3: 2020-01..2025-06 ==')
    r3 = block('same, raw closes', raw, 'same', [T3, T3a, T3b])
    if fix is not None:
        r3f = block('same, with div_fix from 2022-07 (signal inputs and holding returns)', fix, 'same', [T3, T3a, T3b])
        s3 = r3f[(T3[0], 'net')]
        P(f"PASS test 3, old-export part (same sign as test 1, net, with div_fix): {np.sign(s3['mean']) == np.sign(s1['mean'])} "
          f"({s3['mean']:+.3f}%/month); raw closes {r3[(T3[0], 'net')]['mean']:+.3f}%/month")
        block('same500, with div_fix', fix, 'same500', [T3, T3a, T3b], kinds=('net',))
    else:
        P('div_fix version pending')
    block('same500, raw closes', raw, 'same500', [T3, T3a, T3b], kinds=('net',))
    P()
    P('== By period, same-month book (raw closes; 2022-07 on also with div_fix) ==')
    PER = [('2007-05..2009-12', '2007-05', '2009-12'), ('2010-01..2014-12', '2010-01', '2014-12'),
           ('2015-01..2019-12', '2015-01', '2019-12'), ('2020-01..2022-06', '2020-01', '2022-06'),
           ('2022-07..2025-06', '2022-07', '2025-06'), ('all 2007-05..2025-06', '2007-05', '2025-06')]
    block('same', raw, 'same', PER)
    block('same500', raw, 'same500', PER, kinds=('net',))
    if fix is not None:
        block('same, div_fix', fix, 'same', PER[3:], kinds=('net', 'gross'))
    P()
    P('-- by calendar year, same book: net %/month (t), raw | div_fix')
    for y in range(2007, 2026):
        a, b = (f'{y}-01' if y > 2007 else S0), f'{y}-12'
        s = stats(raw, 'same', sel(raw, 'same', a, b))
        extra = ''
        if fix is not None and y >= 2022:
            sf = stats(fix, 'same', sel(fix, 'same', a, b))
            extra = f' | {sf["mean"]:+.3f} ({sf["t"]:+.2f})'
        s5 = stats(raw, 'same500', sel(raw, 'same500', a, b))
        P(f'   {y}: {s["mean"]:+.3f} ({s["t"]:+.2f}), n {s["n"]}{extra};   500 largest {s5["mean"]:+.3f} ({s5["t"]:+.2f})')
    P()
    P('== The contrast: books sorted on the other months of the same years ==')
    P('other = lags 1..120 not multiples of 12 (the plan\'s definition: includes last month\'s reversal and 2-11-month '
      'momentum); other_kln = the same skipping lags 1..11 (Keloharju et al.\'s definition; years 2-10)')
    for nm in ('other', 'other_kln'):
        block(nm + ', raw closes', raw, nm, [T1, T1b, T3, T3b], kinds=('gross', 'net'))
        if fix is not None:
            block(nm + ', div_fix', fix, nm, [T3, T3b], kinds=('gross', 'net'))
    for lab, a, b in (T1, T1b, T3):
        n, m_, t_ = diff_stats(raw, a, b)
        P(f'same minus other_kln (gross), {lab}: {m_:+.3f}%/month, t {t_:+.2f}, n {n}')
    P()
    P('== Legs: the other-month signal (other_kln) of the same book\'s legs, test 1 means ==')
    idx = sel(raw, 'same', S0, '2019-12')
    P(f"same signal long {100 * np.mean(raw['same__sigL'][idx]):+.2f}%/month, short {100 * np.mean(raw['same__sigS'][idx]):+.2f}%; "
      f"other_kln of the long {100 * np.nanmean(raw['same__otherL'][idx]):+.3f}%, of the short {100 * np.nanmean(raw['same__otherS'][idx]):+.3f}%")
    P(f"stocks per leg (test 1): mean {np.mean(raw['same__nL'][idx]):.0f} (min {raw['same__nL'][idx].min()}, "
      f"max {raw['same__nL'][idx].max()}); 500 largest {np.mean(raw['same500__nL'][sel(raw, 'same500', S0, '2019-12')]):.0f}")
    P(f"turnover per month {np.mean(raw['same__turn'][idx]):.2f} of a $1/$1 book (max 4): cost {np.mean(raw['same__cost'][idx]) * 100:.3f}%/month")
    P()
    P('== Excluding January (as both papers also report) ==')
    for lab, a, b in (T1, T3):
        idx = sel(raw, 'same', a, b)
        nj = idx[(months[raw['same__i'][idx]] % 12) != 0]
        for kind in ('gross', 'net'):
            s = stats(raw, 'same', nj, kind)
            P(f'{lab} {kind}: {s["mean"]:+.3f}%/month, t {s["t"]:+.2f}, n {s["n"]}')
        jj = idx[(months[raw['same__i'][idx]] % 12) == 0]
        s = stats(raw, 'same', jj, 'gross')
        P(f'{lab} Januaries only, gross: {s["mean"]:+.3f}%/month, t {s["t"]:+.2f}, n {s["n"]}')
    P()
    P('== Published claim vs this data (gross long-short, %/month (t)) ==')
    for lab, a, b in (('2007-05..2019-12', S0, '2019-12'), ('2020-01..2025-06', '2020-01', '2025-06')):
        bk = fix if (fix is not None and a.startswith('2020')) else raw
        sg = stats(bk, 'same', sel(bk, 'same', a, b), 'gross')
        og = stats(bk, 'other_kln', sel(bk, 'other_kln', a, b), 'gross')
        n, dm, dt = diff_stats(bk, a, b)
        P(f'{lab} ({"div_fix" if bk is fix else "raw"}): same-month {sg["mean"]:+.3f} ({sg["t"]:+.2f}); '
          f'other-month (KLN def.) {og["mean"]:+.3f} ({og["t"]:+.2f}); difference {dm:+.3f} ({dt:+.2f})')
    (HERE / 'results_old.txt').write_text('\n'.join(lines) + '\n')

else:
    # the unseen months: holding months 2025-07..2026-10 (2026-10 partial, through 2026-10-19)
    P('== The unseen months (new export): holding months 2025-07..2026-10; 2026-10 is partial (through 2026-10-19) ==')
    P(f"FF5 + UMD factors: {'loaded' if FAC is not None else 'NOT READY'} (French's daily file ends 2026-08-31, so "
      'alphas cover 2025-07..2026-08)')
    for variant in ('fix', 'raw'):
        bk = B[variant]
        if bk is None:
            continue
        P(f'-- {variant}')
        P(HDR)
        for nm in ('same', 'same500', 'other', 'other_kln'):
            for lab, a, b, part in (('2025-07..2026-10 (16)', '2025-07', '2026-10', True),
                                    ('2025-07..2026-09 (15 complete)', '2025-07', '2026-09', False)):
                idx = sel(bk, nm, a, b, include_partial=part)
                for kind in ('net', 'gross'):
                    row(f'{nm} {lab} {kind}', stats(bk, nm, idx, kind))
        idx = sel(bk, 'same', '2025-07', '2026-10')
        P('   month by month, same book net %: ' + ', '.join(
            f"{month_label(months[i])} {100 * v:+.2f}" for i, v in zip(bk['same__i'][idx], bk['same__net'][idx])))
        n, dm, dt = diff_stats(bk, '2025-07', '2026-10')
        P(f'   same minus other_kln (gross): {dm:+.3f}%/month, t {dt:+.2f}, n {n}')
    (HERE / 'results_new.txt').write_text('\n'.join(lines) + '\n')
