"""Tables from the judge's JSON files (money.py): per-year and summary.

    python3 -I -B money_report.py           -> out/money_table.txt
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, WORK, OUT  # noqa: E402

RUNS = [
    ('FIP score, divfix, uniform', 'score_old_divfix__r21_d0_uniform.json'),
    ('FIP score, divfix, volatility-matched', 'score_old_divfix__r21_d0_volatility.json'),
    ('FIP score, plain, uniform', 'score_old_plain__r21_d0_uniform.json'),
    ('FIP score, plain, volatility-matched', 'score_old_plain__r21_d0_volatility.json'),
    ('Baseline: plain momentum, divfix, uniform', 'score_mom_old_divfix__r21_d0_uniform.json'),
    ('Baseline: plain momentum, divfix, volatility-matched', 'score_mom_old_divfix__r21_d0_volatility.json'),
    ('UNSEEN FIP score, divfix, uniform', 'score_new_divfix__r21_d0_uniform.json'),
    ('UNSEEN FIP score, divfix, volatility-matched', 'score_new_divfix__r21_d0_volatility.json'),
    ('UNSEEN FIP score, plain, uniform', 'score_new_plain__r21_d0_uniform.json'),
    ('UNSEEN FIP score, plain, volatility-matched', 'score_new_plain__r21_d0_volatility.json'),
]


def main():
    lines = []
    say = lines.append
    say('| Run | Change over random, 0 bps (se, ahead) | 5 bps | 20 bps | Years beating random at p < 0.1 (0/5/20 bps) | '
        'Picks mean yearly return, 0 / 20 bps | Random median, 0 / 20 bps | IC z (actual returns) |')
    say('|---|---|---|---|---|---|---|---|')
    per_year = {}
    for label, f in RUNS:
        p = WORK / f
        if not p.exists():
            continue
        js = json.loads(p.read_text())
        c = [c2eval.change_over_random(js, i) for i in range(3)]
        if js.get('years'):
            s = js['summary']
            beat = '/'.join(str(b['years']) for b in s['beat_random_p_lt_0_1'])
            ent = [y['costs'] for y in js['years']]
            pk0 = np.mean([e[0]['final_pct'] for e in ent])
            pk2 = np.mean([e[2]['final_pct'] for e in ent])
            rm0 = np.mean([e[0]['random_median_pct'] for e in ent])
            rm2 = np.mean([e[2]['random_median_pct'] for e in ent])
            ic = f"{s['ic_ret_z']:+.2f}"
            per_year[label] = [(y['start'][:10], y['costs'][0]['final_pct'], y['costs'][0]['random_median_pct'],
                                y['costs'][0]['p']) for y in js['years']]
            say(f'| {label} | {c[0]["change"]:+.1f} ({c[0]["se"]:.1f}, {c[0]["ahead"]}/{c[0]["years"]}) | '
                f'{c[1]["change"]:+.1f} | {c[2]["change"]:+.1f} | {beat} of {s["years"]} | {pk0:+.1f}% / {pk2:+.1f}% | '
                f'{rm0:+.1f}% / {rm2:+.1f}% | {ic} |')
        else:
            st = js['report']['strategy']
            ps = '/'.join(f"{x['p']:.2f}" for x in st)
            say(f'| {label} | {c[0]["change"]:+.1f} | {c[1]["change"]:+.1f} | {c[2]["change"]:+.1f} | p = {ps} | '
                f'{st[0]["final_pct"]:+.1f}% / {st[2]["final_pct"]:+.1f}% | {st[0]["random_median_pct"]:+.1f}% / '
                f'{st[2]["random_median_pct"]:+.1f}% | |')
    say('')
    labs = [l for l in per_year]
    say('Per year, 0 bps: picks / random median (p)')
    say('| Year from | ' + ' | '.join(labs) + ' |')
    say('|---|' + '---|' * len(labs))
    for i in range(len(per_year[labs[0]])):
        row = [per_year[l][i] for l in labs]
        say(f'| {row[0][0]} | ' + ' | '.join(f'{r[1]:+.1f}% / {r[2]:+.1f}% ({r[3]:.2f})' for r in row) + ' |')
    (OUT / 'money_table.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
