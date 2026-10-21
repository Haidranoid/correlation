"""Judge-based files: mean over the graded years of (the judge account's 5-bps yearly return - SPY's Yahoo total
return over the same judge year). Writes judge_based.json and judge_based.txt next to this script."""
import json
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/after_costs/group_c')
import ac_common as A  # noqa: E402

W = A.WORK
FILES = {
    '-1.6-overnight-peers.md': W / 'overnight_peers/scores_on_old__r21_d0_uniform.json',
    '+4.6-hidden-text-peers.md': W / 'hidden_text_peers/scores/hidden_text_peers_old__r21_d0_uniform.json',
    '-3.8-factor-momentum.md': W / 'factor_momentum/book_old__r21_d0_uniform.json',
    '+0.0-short-volume.md': W / 'short_volume/scores_G_old__r5_d0_uniform.json',
    '-3.5-theme-joiners.md': W / 'theme_joiners/scores_old__r21_d0_uniform.json',
    '-10.2-decoupling.md': W / 'decoupling/scores_old__r5_d0_uniform.json',
    '-0.6-announcement-beta.md': W / 'announcement_beta/scores/beta_timing_old__r1_d0_uniform.json',
    '-0.2-seasonality.md': W / 'seasonality/scores/season_old_raw__r1_d0_uniform.json',
    '-4.4-betting-against-beta.md': W / 'betting_against_beta/bab_money_old__r21_d0_uniform.json',
    '-7.4-52-week-high.md': W / 'week52_high/pth_money_old__r21_d0_uniform.json',
    '-6.8-etf-stat-arb.md': W / 'etf_stat_arb/money_old__r1_d0_uniform.json',
    '-3.6-max-lottery.md': W / 'max_lottery/max_money_old__r21_d0_uniform.json',
    '+2.0-frog-in-the-pan.md': W / 'frog_in_the_pan/score_old_divfix__r21_d0_uniform.json',
    '+2.0-frog-in-the-pan.md (plain)': W / 'frog_in_the_pan/score_old_plain__r21_d0_uniform.json',
}

out, lines = {}, []
for f, p in FILES.items():
    js, rows = A.judge_rows(p, 5.0)
    s = A.summarize(rows)
    out[f] = {'json': str(p), 'rows': rows, **s}
    lines.append(f"{f}  {p.name}")
    lines.append(f"  years {rows[0]['first_day']}..{rows[-1]['last_day']} ({s['n']} judge years); change over random at 0 bps "
                 f"{s['change0']:+.2f} (file name check)")
    lines.append(f"  account at 5 bps {s['acct_mean']:+.2f}%/yr; SPY total return {s['spy_mean']:+.2f}%/yr; "
                 f"after-cost number {s['after_cost']:+.2f} -> {A.fmt(s['after_cost'])}")
    for r in rows:
        lines.append(f"    {r['first_day']}..{r['last_day']}  acct0 {r['acct0']:+8.2f}  acct5 {r['acct']:+8.2f}  "
                     f"SPY(yahoo TR) {r['spy']:+7.2f}  SPY(judge) {r['judge_spy']:+7.2f}  diff {r['acct'] - r['spy']:+8.2f}")
(A.OUT / 'judge_based.json').write_text(json.dumps(out, indent=1))
(A.OUT / 'judge_based.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
