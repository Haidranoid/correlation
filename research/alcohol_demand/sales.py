"""C4: are alcohol sales themselves forecastable? On calendar months m, forecast the 12-month growth of beer, wine and
liquor store sales three months ahead, g(m + 3), from month m's information: (a) the latest growth g(m) alone (an
AR(1) in the growth rate); (b) the average of six fits, each g(m) plus one input's month-m value. Out-of-sample R^2
against the expanding mean. Final-vintage (revised) sales, so reported only.

    python3 -I -B sales.py old     # -> sales_old.txt, sales_old.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402

AHEAD = 3


def main(tag):
    cut = K.mindex(K.CUT[tag])
    S = K.read_fred('MRTSSM4453USS')
    g = K.yoy(S)
    X = K.monthly_inputs()
    months = np.array([m for m in range(K.mindex('1993-01'), cut + 1)])
    gm = np.array([g.get(K.mkey(m), np.nan) for m in months])
    ga = np.array([g.get(K.mkey(m + AHEAD), np.nan) if m + AHEAD <= cut else np.nan for m in months])
    Xm = np.column_stack([[X[n].get(K.mkey(m), np.nan) for m in months] for n in K.INPUTS])
    ok = np.isfinite(gm) & np.isfinite(ga) & np.isfinite(Xm).all(1)
    n = len(months)
    f0, fa, fb = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(gm[i]) and np.isfinite(Xm[i]).all()):
            continue
        el = np.nonzero(ok[:i] & (months[:i] + AHEAD <= months[i]))[0]
        if len(el) < K.MIN_FIT:
            continue
        yy = ga[el]
        f0[i] = yy.mean()
        A = np.column_stack([np.ones(len(el)), gm[el]])
        b, *_ = np.linalg.lstsq(A, yy, rcond=None)
        fa[i] = b[0] + b[1] * gm[i]
        fk = []
        for k in range(Xm.shape[1]):
            A = np.column_stack([np.ones(len(el)), gm[el], Xm[el, k]])
            b, *_ = np.linalg.lstsq(A, yy, rcond=None)
            fk.append(b[0] + b[1] * gm[i] + b[2] * Xm[i, k])
        fb[i] = np.mean(fk)
    keys = np.array([K.mkey(m) + '-15' for m in months])
    lines, res = [], {}
    for pn, (lo, hi) in (('2008-2019', ('2008-01-01', '2019-12-31')), ('2020-2025', ('2020-01-01', K.CUT[tag]))):
        msk = K.sel(keys, lo, hi)
        ea = K.evaluate(ga, f0, fa, msk, boot=False)
        eb = K.evaluate(ga, f0, fb, msk, boot=False)
        ec = K.evaluate(ga, fa, fb, msk, boot=False)     # do the inputs add to the latest growth?
        res[pn] = {'latest_growth': ea, 'with_inputs': eb, 'inputs_over_latest': ec}
        s = (f'{pn}: forecasting sales growth {AHEAD} months ahead: latest growth alone R2 {ea["r2os"]:+.3f} '
             f'(n {ea["n"]}); with the inputs R2 {eb["r2os"]:+.3f}; the inputs over the latest growth alone: '
             f'R2 {ec["r2os"]:+.3f}, Clark-West t {ec["cw_t"]:+.2f}')
        print(s)
        lines.append(s)
    (K.HERE / f'sales_{tag}.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / f'sales_{tag}.json')


if __name__ == '__main__':
    main(sys.argv[1])
