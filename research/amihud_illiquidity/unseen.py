"""The 16 unseen months (2025-06-30..2026-10-19), once, at the end, with the frozen code: check the hashes, check that
the new export's overlap equals the old one, run every script on the new export, compare the old span across the
two exports, and summarize.

    python3 -I -B unseen.py   -> out/unseen.txt (and every *_new output)
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, HERE, OUT, WORK, OLD_T, log  # noqa: E402

SCRIPTS = ['common.py', 'check_splits.py', 'panels.py', 'tests.py', 'timeseries.py', 'money.py', 'sanity_reversal.py',
           'unseen.py', 'forward.py']


def check_hashes():
    want = {}
    for l in (HERE / 'frozen_sha256.txt').read_text().splitlines():
        if l.strip() and not l.startswith('#'):
            h, f = l.split()
            want[f] = h
    bad = [f for f in SCRIPTS if hashlib.sha256((HERE / f).read_bytes()).hexdigest() != want.get(f)]
    return bad


def overlap():
    Eo = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close', 'dollar_volume'))
    En = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('close', 'dollar_volume'))
    assert np.array_equal(Eo['days'], En['days'][:OLD_T])
    rn = {t: j for j, t in enumerate(En['tickers'])}
    same_c = same_v = common = 0
    diff = []
    for j, t in enumerate(Eo['tickers']):
        k = rn.get(t)
        if k is None:
            continue
        common += 1
        a = np.array_equal(np.asarray(Eo['close'][j]), np.asarray(En['close'][k, :OLD_T]), equal_nan=True)
        b = np.array_equal(np.asarray(Eo['dollar_volume'][j]), np.asarray(En['dollar_volume'][k, :OLD_T]), equal_nan=True)
        same_c += a
        same_v += b
        if not (a and b) and len(diff) < 15:
            diff.append(str(t))
    return dict(old_stocks=len(Eo['tickers']), new_stocks=len(En['tickers']), common=common, same_close=same_c,
                same_dv=same_v, examples=diff)


def run(script, *args):
    log('run', script, *args)
    env = dict(os.environ)
    r = subprocess.run([sys.executable, '-I', '-B', str(HERE / script), *args], cwd=HERE, env=env,
                       capture_output=True, text=True)
    (OUT / f'unseen_{Path(script).stem}_{"_".join(args)}.log').write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f'{script} {args} failed:\n{r.stderr[-3000:]}')


def main():
    L = ['# The unseen months, run once with the frozen code\n']
    bad = check_hashes()
    L.append(f'Hashes before the run: {"all match" if not bad else "MISMATCH " + str(bad)}')
    if bad:
        (OUT / 'unseen.txt').write_text('\n'.join(L) + '\n')
        raise SystemExit('frozen hashes do not match: ' + str(bad))
    ov = overlap()
    L.append(f"Overlap: old export {ov['old_stocks']} stocks, new {ov['new_stocks']}; {ov['common']} tickers in both. Over "
             f"the old export's {OLD_T} days, closes identical for {ov['same_close']} and dollar volume for {ov['same_dv']}. "
             f"Examples that differ: {', '.join(ov['examples'])}")
    for s, a in (('check_splits.py', ['new']), ('panels.py', ['new']), ('tests.py', ['new']), ('timeseries.py', ['new']),
                 ('money.py', ['scores', 'new']), ('money.py', ['judge', 'new']), ('money.py', ['costs', 'new'])):
        run(s, *a)
    # the old span on both exports
    so, sn = np.load(WORK / 'series_old.npz'), np.load(WORK / 'series_new.npz')
    k = int(so['complete'].sum())
    for key in ('P1_coef', 'P2_coef', 'P1_book_resid_net', 'P2_book_resid_net', 'P1_book_raw_net', 'P2_book_raw_net'):
        a, b = so[key][:len(so['complete'])], sn[key][:len(so['complete'])]
        m = so['complete'] & np.isfinite(a) & np.isfinite(b)
        L.append(f'Old span, {key}: correlation old vs new export {np.corrcoef(a[m], b[m])[0, 1]:.4f} over {int(m.sum())} '
                 f'formations; means {100 * a[m].mean():+.3f} vs {100 * b[m].mean():+.3f}')
    pan = (OUT / 'panels_new.txt').read_text().splitlines()
    L += [l for l in pan if l.startswith(('Days with', 'By year', 'Check'))]
    L.append('Results: out/results_new.txt, out/timeseries_new.txt, out/money_new.txt, out/money_costs_new.json')
    bad = check_hashes()
    L.append(f'Hashes after the run: {"all match" if not bad else "MISMATCH " + str(bad)}')
    (OUT / 'unseen.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
