"""Forward confirmation: run a later download through the frozen code and keep a ledger of first-seen results.

    python3 -I -B forward.py EXPORT_DIR STORE_DIR   -> out/forward_ledger.json, out/forward_report.txt

EXPORT_DIR is a judge export (meta.json, days.npy, close.npy, dollar_volume.npy, symbols.tsv) built from a download
after 2026-10-19; STORE_DIR the matching price store (for the twins and the fund-like flag). The download has no
dividend correction, so forward returns are price-only. A formation already in the ledger keeps its first-seen values
(revisions in later downloads are ignored). A download more than 31 calendar days after the previous one is logged as
a gap, never filled. Report at 12 forward formations; P1 and P2 are judged at 36 (Holm, t >= 2, as in the plan).
The judge account isn't run forward (it needs a judge sandbox for the download).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, HERE, OUT, WORK, FORWARD_FROM, log, mean_t, norm_sf, holm  # noqa: E402

LEDGER = OUT / 'forward_ledger.json'


def main(export_dir, store_dir):
    env = dict(os.environ, AMIHUD_FWD_EXPORT=str(export_dir), AMIHUD_FWD_STORE=str(store_dir))
    for s in ('panels.py', 'tests.py', 'timeseries.py'):
        log('run', s)
        r = subprocess.run([sys.executable, '-I', '-B', str(HERE / s), 'fwd'], cwd=HERE, env=env, capture_output=True,
                           text=True)
        (OUT / f'forward_{Path(s).stem}.log').write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f'{s} failed:\n{r.stderr[-3000:]}')
    meta = json.loads((Path(export_dir) / 'meta.json').read_text())
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'downloads': [], 'gaps': [], 'formations': {}}
    last = meta['last_day']
    if led['downloads']:
        prev = led['downloads'][-1]['last_day']
        if (np.datetime64(last) - np.datetime64(prev)).astype(int) > 31:
            led['gaps'].append({'after': prev, 'next': last})
    led['downloads'].append({'export': str(export_dir), 'last_day': last})
    Z = np.load(WORK / 'form_fwd.npz')
    S = np.load(WORK / 'series_fwd.npz')
    first, comp = Z['first_ret'], Z['complete']
    for k in np.nonzero(comp & (first >= FORWARD_FROM))[0]:
        key = str(Z['iso'][k])
        if key in led['formations']:
            continue
        led['formations'][key] = {c: float(S[c][k]) for c in ('P1_coef', 'P2_coef', 'P1_book_resid_net', 'P2_book_resid_net')}
        led['formations'][key]['first_seen_in'] = last
    LEDGER.write_text(json.dumps(led, indent=1))
    F = led['formations']
    n = len(F)
    L = [f'Forward formations in the ledger: {n} (downloads {len(led["downloads"])}, gaps {len(led["gaps"])})']
    if n >= 12:
        pv = {}
        for pk in ('P1', 'P2'):
            c = np.array([v[f'{pk}_coef'] for v in F.values()])
            b = np.array([v[f'{pk}_book_resid_net'] for v in F.values()])
            m, t, tnw, nn = mean_t(c)
            pv[pk] = norm_sf(tnw)
            L.append(f'{pk}: coefficient {100 * m:+.3f}% (t NW {tnw:+.2f}), controlled book net {100 * np.nanmean(b):+.3f}%, '
                     f'{nn} formations')
        if n >= 36:
            H = holm(pv)
            L.append('Judged at 36: ' + '; '.join(f'{k} p {v[0]:.4f} vs {v[1]:.4f}: {"pass" if v[2] else "fail"}'
                                                  for k, v in H.items()))
    (OUT / 'forward_report.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
