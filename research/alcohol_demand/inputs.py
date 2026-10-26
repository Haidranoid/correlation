"""Build the decisions, inputs and targets for one tag and check the data (no forecasts, no relationships).

    python3 -I -B inputs.py old     # -> WORK/inputs_old.npz, inputs_old.txt
    python3 -I -B inputs.py new     # the unseen months (run once, at the end)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402

TARGETS = {                       # name -> (basket, benchmark)
    'ALC-SPY': (K.ALC, 'SPY'),
    'ALC': (K.ALC, None),
    'ALC-XLP': (K.ALC, 'XLP'),
    'GLOBAL-SPY': (K.GLOBAL, 'SPY'),
    'XLP-SPY': (('XLP',), 'SPY'),
    'XLY-SPY': (('XLY',), 'SPY'),
    'QQQ-SPY': (('QQQ',), 'SPY'),
    'SPY': (('SPY',), None),
    'WIDE-SPY': (K.ALC + K.GLOBAL, 'SPY'),
    'GLOBAL': (K.GLOBAL, None),
    'XLP': (('XLP',), None),
    'XLY': (('XLY',), None),
    'QQQ': (('QQQ',), None),
}


def main(tag):
    lines = []

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    axis, off, ed, P, fills = K.price_panel(tag)
    # the axis equals the export's days from 2001-12-31
    seg = axis[off:off + len(ed)]
    same = len(seg) == len(ed) and np.array_equal(seg, ed)
    out(f'{tag}: axis {K.iso(axis[0])}..{K.iso(axis[-1])} ({len(axis)} days); export {K.iso(ed[0])}..{K.iso(ed[-1])} '
        f'({len(ed)} days); offset {off}; axis == export days from the export start: {same}')
    if not same:
        a, b = set(seg.tolist()), set(ed.tolist())
        out(f'  axis-only days {len(a - b)}, export-only days {len(b - a)}: '
            f'{[K.iso(x) for x in sorted(a ^ b)][:10]}')
    for t in K.TICKERS:
        x = P[t]
        first = int(np.argmax(np.isfinite(x)))
        out(f'  {t:5s} first {K.iso(axis[first])}  filled days {fills[t]}')
    D = K.decision_days(len(axis), off)
    dates = K.iso_days(axis[D])
    out(f'decisions: {len(D)} from {dates[0]} to {dates[-1]} (hold ends {K.iso(axis[D[-1] + 1 + K.H])})')
    assert np.all(np.diff(D) == K.H)

    X = K.monthly_inputs()
    cols = {}
    for caution in (0, 1):
        for name in K.INPUTS + ('R',):
            v = np.full(len(D), np.nan)
            used = []
            for i, d in enumerate(dates):
                v[i], k = K.input_at(X, name, d, caution)
                used.append(k)
            cols[f'{name}_c{caution}'] = v
            if caution == 0:
                fin = np.isfinite(v)
                first = dates[np.argmax(fin)]
                late = [(d, u) for d, u in zip(dates, used)
                        if u is not None and K.mindex(u) != K.usable_month(name, d, caution)]
                out(f'  input {name}: first decision with a value {first}; decisions without one after it '
                    f'{int((~fin[np.argmax(fin):]).sum())}; decisions using an earlier month than the rule '
                    f'(late or missing release) {len(late)} {late[:4]}')
    # examples of the month used, for reading the rule
    for d in ('2008-01-02', '2013-11-04', '2020-03-16', '2025-11-12'):
        j = np.searchsorted(dates, d)
        if j < len(dates):
            out(f'  decision {dates[j]}: ' + ', '.join(f'{n} {K.input_at(X, n, dates[j])[1]}'
                                                      for n in K.INPUTS + ('R',)))

    Y = {}
    for name, (names, bench) in TARGETS.items():
        Y[name] = K.target(P, names, D, bench)
        fin = np.isfinite(Y[name])
        out(f'  target {name}: decisions with a value {int(fin.sum())}, first {dates[np.argmax(fin)]}')
    nalc = np.array([K.basket_log(P, K.ALC, t0 + 1, t0 + 1 + K.H, need_at=t0)[1] for t0 in D])
    out(f'  ALC names per decision from 2005: min {nalc[dates >= "2005-01-01"].min()} '
        f'max {nalc[dates >= "2005-01-01"].max()}')
    six = np.column_stack([cols[f'{n}_c0'] for n in K.INPUTS])
    allok = np.isfinite(six).all(1) & np.isfinite(Y['ALC-SPY'])
    out(f'  decisions with all six inputs and the primary target: {int(allok.sum())}, first {dates[np.argmax(allok)]}')
    for lo, hi, nm in ((K.DEV[0], K.DEV[1], 'development'), (K.CHECK[0], K.CHECK[1], 'check'),
                       (K.UNSEEN[0], K.UNSEEN[1], 'unseen')):
        out(f'  {nm}: {int((K.sel(dates, lo, hi) & allok).sum())} decisions')

    np.savez(K.WORK / f'inputs_{tag}.npz', axis=axis, off=off, D=D, dates=dates,
             **{f'x_{k}': v for k, v in cols.items()}, **{f'y_{k}': v for k, v in Y.items()},
             **{f'P_{t}': P[t] for t in K.TICKERS})
    (K.HERE / f'inputs_{tag}.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
