"""Fidelity check, before any returns are read: the top-20 pairs at a few formations (tickers, SSD, sigma, trigger)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

tag, source, div = (sys.argv[1:4] + ['old', 'export', 'nodiv'][len(sys.argv[1:4]):])[:3]
E = c2eval.load_export(C.EXPORTS[tag], arrays=())
R = np.load(C.WORK / f'run_{tag}_{source}_{div}_all.npz')
rebal, days = R['rebal'], R['days']
lines = []
for iso_ in ('2008-06-30', '2015-06-30', '2022-06-30'):
    k = int(np.searchsorted(days[rebal], (np.datetime64(iso_) - np.datetime64('1970-01-01')).astype(int)))
    lines.append(f"\nFormation ending {c2eval.iso(days[rebal[k]])} (k={k}), universe {R['nuni'][k]} stocks; top 20 pairs:")
    lines.append(f"{'#':>2} {'pair':<14} {'SSD':>7} {'sigma':>7} {'2 sigma':>8} {'opens':>5} {'converged':>9}")
    for p in range(C.NPAIRS):
        a, b = R['pairs'][k, p]
        lines.append(f"{p + 1:>2} {E['tickers'][a] + '/' + E['tickers'][b]:<14} {R['ssd'][k, p]:7.4f} {R['sigma'][k, p]:7.4f} "
                     f"{2 * R['sigma'][k, p]:8.4f} {R['n_open'][k, p]:>5} {R['n_conv'][k, p]:>9}")
v = R['valid']
lines.append(f"\nAll formations: mean 2-sigma trigger {2 * np.nanmean(R['sigma'][v]):.4f} (GGR top 20: 0.0528); "
             f"median SSD of pair 1 {np.nanmedian(R['ssd'][v, 0]):.4f}, of pair 20 {np.nanmedian(R['ssd'][v, -1]):.4f}")
# how often a stock appears in more than one of the 20 pairs
rep = [C.NPAIRS * 2 - len(np.unique(R['pairs'][k])) for k in np.nonzero(v)[0]]
lines.append(f"Stocks appearing in more than one pair: mean {np.mean(rep):.1f} repeats per formation")
from collections import Counter  # noqa: E402
cnt = Counter(E['tickers'][x] for k in np.nonzero(v)[0] for x in R['pairs'][k].ravel())
lines.append('Most frequent stocks in the top 20 (formations x pairs): ' + ', '.join(f'{t} {n}' for t, n in cnt.most_common(25)))
txt = '\n'.join(lines)
print(txt)
(C.OUT / f'examples_{tag}_{source}_{div}.txt').write_text(txt + '\n')
