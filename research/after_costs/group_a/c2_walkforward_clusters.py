"""The cluster lead-lag walk-forward (`c2 census holdout --family clusters --walk-forward`, paper settings) at
0/5/20 bps, by the same cut-store method as c2_walkforward_costs.py.

research/cluster_leadlag_walkforward.txt was made on 2026-10-20 after the default store became stooq-2026-10-19,
and its 2024-25 row (-11.1% against a random median of +1.8%) is reproduced exactly on that store cut before
2025-06-28 (and not on stooq-2025-06-27), so this uses stooq-2026-10-19, with census main-2026-10-19 (the
default census then; the cluster family doesn't use census pairs, and `--name main` gives the same result).
Writes runs/clusters_<year>.txt and c2_walkforward_clusters.json.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path('/home/bhoult/programming/correlation/research/after_costs/group_a')
sys.path.insert(0, str(HERE))
import c2_walkforward_costs as base  # noqa: E402

STORE = Path.home() / '.local/share/correlation2/stores/stooq-2026-10-19'


def run(store, year):
    cmd = [base.C2, '--store', str(store), 'census', 'holdout', '--name', 'main-2026-10-19', '--family', 'clusters',
           '--start', f'{year}-06-28', '--costs', '0,5,20']
    r = subprocess.run(cmd, capture_output=True, text=True, env={'RAYON_NUM_THREADS': '8', 'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin'})
    (base.RUNS / f'clusters_{year}.txt').write_text(' '.join(cmd[1:]) + '\n\n' + r.stdout + r.stderr)
    if r.returncode != 0:
        return {'error': (r.stderr or r.stdout)[-500:]}
    return base.parse_holdout(r.stdout)


def main():
    base.RUNS.mkdir(exist_ok=True)
    wf = base.parse_walkforward(HERE.parent.parent / 'cluster_leadlag_walkforward.txt')
    out = {}
    for year in range(2006, 2025):
        row = wf.get(str(year))
        if row is None:
            continue
        cut = base.CUTS / f'new_cut_{year + 1}-06-28'
        if not (cut / 'prices.bin').exists():
            subprocess.run([sys.executable, '-I', '-B', str(HERE / 'truncate_store.py'), str(STORE), str(cut), f'{year + 1}-06-28'], check=True)
        res = run(cut, year)
        res['walkforward_row'] = row
        if 'costs' in res:
            res['matches_walkforward_0bps'] = (abs(round(res['costs'][0.0]['pct'], 1) - row['picks0']) < 0.051
                                               and abs(round(res['costs'][0.0]['random_median_pct'], 1) - row['random0']) < 0.051
                                               and res['first_day'] == row['first_day'] and res['last_day'] == row['last_day'])
        out[str(year)] = res
        c = res.get('costs', {})
        print(f"clusters {year} wf {row['picks0']:+6.1f}/{row['random0']:+6.1f}  run "
              + (f"{c[0.0]['pct']:+6.1f}/{c[0.0]['random_median_pct']:+6.1f}  5bps {c[5.0]['pct']:+6.1f}  match {res['matches_walkforward_0bps']}" if c else res.get('error', '')), flush=True)
        shutil.rmtree(cut)
    (HERE / 'c2_walkforward_clusters.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
