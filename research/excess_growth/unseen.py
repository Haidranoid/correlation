"""The 16 unseen months (2025-06-30..2026-10-19), run once at the end with the frozen code.

    python3 -I -B unseen.py

Checks the frozen hashes, that the new export's first 5,911 days equal the old export's, and how the closes compare
over the overlap; then run.py new, funds.py new and after_costs.py new. Writes unseen.txt.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402

np = C.np


def check_hashes():
    bad = []
    for line in (C.HERE / 'frozen_sha256.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        h, name = line.split()
        if hashlib.sha256((C.HERE / name).read_bytes()).hexdigest() != h:
            bad.append(name)
    return bad


def main():
    L = []
    bad = check_hashes()
    L.append(f'Frozen hashes before the run: {"all match" if not bad else "MISMATCH " + str(bad)}')
    assert not bad
    Eo = C.c2eval.load_export(C.EXPORTS['old'], arrays=('close',))
    En = C.c2eval.load_export(C.EXPORTS['new'], arrays=('close',))
    To = len(Eo['days'])
    same_days = bool(np.array_equal(En['days'][:To], Eo['days']))
    L.append(f"New export: {En['close'].shape}, first {To} days equal the old export's: {same_days}")
    assert same_days
    # closes over the overlap, for tickers in both exports
    no = {t: j for j, t in enumerate(Eo['tickers'])}
    pairs = [(no[t], j) for j, t in enumerate(En['tickers']) if t in no]
    io = np.array([a for a, _ in pairs])
    jn = np.array([b for _, b in pairs])
    co = np.asarray(Eo['close'])[io]
    cn = np.asarray(En['close'])[jn][:, :To]
    both = np.isfinite(co) & np.isfinite(cn)
    with np.errstate(all='ignore'):
        rel = np.abs(cn / co - 1)
    eq = (both & (rel < 1e-6)).sum()
    L.append(f"Tickers in both exports: {len(pairs)}; stock-days with a close in both: {int(both.sum())}; equal to 1e-6: "
             f"{eq / both.sum():.4%}; differing by more than 1%: {int((both & (rel > 0.01)).sum())} "
             f"(re-adjustments of the history in the later download)")
    del co, cn
    import run as RUN
    import funds as F
    import after_costs as AC
    res = RUN.main('new')
    F.main('new')
    AC.main('new')
    t = res['tests']['unseen']
    L.append('\nUnseen months, rebalanced minus buy-and-hold (log points a year):')
    for k in ('P1', 'P2', 'P1 gross', 'P2 gross', 'delay 0 daily gross'):
        L.append(f"  {k:22s} {100 * t[k]['per_year']:+.3f}%  NW t {t[k]['t_nw21']:+.2f}  p {t[k]['p']:.4f}  ({t[k]['days']} days)")
    acn = json.loads((C.HERE / 'ac_new.json').read_text())
    L.append(f"After costs, the unseen months (mean of the two partial calendar years): {acn['after_costs']:+.2f}; "
             f"buy-and-hold {acn['after_costs_B1']:+.2f}; before costs vs random {acn['before_vs_random']:+.2f}")
    bad = check_hashes()
    L.append(f'Frozen hashes after the run: {"all match" if not bad else "MISMATCH " + str(bad)}')
    txt = '\n'.join(L) + '\n'
    (C.HERE / 'unseen.txt').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
