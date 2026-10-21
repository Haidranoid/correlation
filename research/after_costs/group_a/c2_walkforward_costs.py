"""Recompute c2's walk-forward years (census pairs, peer momentum, 5-day reversal) at 0/5/20 bps.

`c2 census holdout --walk-forward` grades each one-year holdout at 0 bps only (crates/c2-cli/src/main.rs hard-codes
costs_bps [0.0] there). A single holdout grades every cost level but runs to the store's last day. So for each
walk-forward year y (June 28 of y to June 28 of y+1, the walk-forward's own bounds) this cuts a copy of the store the
walk-forward used (stooq-2025-06-27) before June 28 of y+1 and runs the single holdout from June 28 of y on it.
The 0-bps picks and random median must equal the walk-forward row in research/*.txt; the script records both.
The last year (2024-25) ends at the store's own last day, so it runs on the uncut store.

Writes runs/<family>_<year>.txt and c2_walkforward_costs.json. The cut stores go to
~/.local/share/correlation2/experiments/after_costs/group_a/stores/ and are deleted after use.
Run: systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -- python3 -I -B c2_walkforward_costs.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path('/home/bhoult/programming/correlation/research/after_costs/group_a')
C2 = '/home/bhoult/programming/correlation/target/release/c2'
STORE = Path.home() / '.local/share/correlation2/stores/stooq-2025-06-27'
CUTS = Path.home() / '.local/share/correlation2/experiments/after_costs/group_a/stores'
RUNS = HERE / 'runs'
FAMILIES = {'census': ['--family', 'census'], 'peers': ['--family', 'peers'],
            'reversal5': ['--family', 'reversal', '--lookback', '5']}
WF = {'census': 'census_main_walkforward.txt', 'peers': 'peer_momentum_walkforward.txt',
      'reversal5': 'walkforward_reversal_5.txt'}


def parse_holdout(text):
    out = {'costs': {}}
    m = re.search(r'Holdout (\S+) \.\. (\S+) \((\d+) trading days\)', text)
    out['first_day'], out['last_day'], out['days'] = m.group(1), m.group(2), int(m.group(3))
    m = re.search(r'Buy and hold SPY \(S&P 500\)\s+(\d+)\s+(-?[\d.]+)%', text)
    out['spy_store_pct'] = 100 * (int(m.group(1)) / 10000 - 1)
    for label, val, med, p in re.findall(r'Picks, (no trading costs|[\d.]+ bps per trade)\s+(\d+)\s+-?[\d.]+%\s+random guessers: median (\d+), [\d.]+% did at least as well \(p = ([\d.]+)\)', text):
        c = 0.0 if label.startswith('no') else float(label.split()[0])
        out['costs'][c] = {'final': int(val), 'pct': 100 * (int(val) / 10000 - 1), 'random_median_pct': 100 * (int(med) / 10000 - 1), 'p': float(p)}
    return out


def parse_walkforward(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'(\d{4}-\d\d-\d\d)\.\.(\d{4}-\d\d-\d\d)\s+\d+\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+(-?[\d.]+)%\s+([\d.]+)', line)
        if m:
            rows[m.group(1)[:4]] = {'first_day': m.group(1), 'last_day': m.group(2), 'picks0': float(m.group(5)),
                                    'random0': float(m.group(6)), 'p0': float(m.group(7))}
    return rows


def run_c2(store, fam, year):
    cmd = [C2, '--store', str(store), 'census', 'holdout', '--name', 'main', *FAMILIES[fam],
           '--start', f'{year}-06-28', '--costs', '0,5,20']
    r = subprocess.run(cmd, capture_output=True, text=True, env={'RAYON_NUM_THREADS': '8', 'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin'})
    (RUNS / f'{fam}_{year}.txt').write_text(' '.join(cmd[1:]) + '\n\n' + r.stdout + r.stderr)
    if r.returncode != 0:
        return {'error': (r.stderr or r.stdout)[-500:]}
    return parse_holdout(r.stdout)


def main():
    RUNS.mkdir(exist_ok=True)
    CUTS.mkdir(parents=True, exist_ok=True)
    wf = {f: parse_walkforward(HERE.parent.parent / p) for f, p in WF.items()}
    out = {f: {} for f in FAMILIES}
    for year in range(2006, 2025):
        if year < 2024:
            cut = CUTS / f'old_cut_{year + 1}-06-28'
            if not (cut / 'prices.bin').exists():
                subprocess.run([sys.executable, '-I', '-B', str(HERE / 'truncate_store.py'), str(STORE), str(cut), f'{year + 1}-06-28'], check=True)
            store = cut
        else:
            cut, store = None, STORE
        for fam in FAMILIES:
            row = wf[fam].get(str(year))
            if row is None:
                continue
            res = run_c2(store, fam, year)
            res['walkforward_row'] = row
            if 'costs' in res:
                res['matches_walkforward_0bps'] = (abs(round(res['costs'][0.0]['pct'], 1) - row['picks0']) < 0.051
                                                   and abs(round(res['costs'][0.0]['random_median_pct'], 1) - row['random0']) < 0.051
                                                   and res['first_day'] == row['first_day'] and res['last_day'] == row['last_day'])
            out[fam][str(year)] = res
            c = res.get('costs', {})
            print(f"{fam:9s} {year} wf {row['picks0']:+6.1f}/{row['random0']:+6.1f}  run "
                  + (f"{c[0.0]['pct']:+6.1f}/{c[0.0]['random_median_pct']:+6.1f}  5bps {c[5.0]['pct']:+6.1f}  match {res['matches_walkforward_0bps']}" if c else res.get('error', '')), flush=True)
        if cut is not None:
            shutil.rmtree(cut)
    (HERE / 'c2_walkforward_costs.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
