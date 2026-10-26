"""The plan's split check, run before any ILLIQ is used: on 20 known splits (the list of an earlier test's check_splits.py), is
the export's dollar volume continuous across the split day? If Stooq adjusted closes but not volumes, dollar volume
before the split would read low by the split ratio, and ILLIQ high.

    python3 -I -B check_splits.py old|new   -> out/check_splits_<tag>.txt

A split passes if the 20-day median dollar volume after the split day over the 20 days before is nearer 1 than the
split ratio in logs, and the close shows no jump of half the split ratio's log on the day.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import c2eval, np, OUT, EXPORTS, log  # noqa: E402

SPLITS = [('AAPL', '2014-06-09', 7), ('AAPL', '2020-08-31', 4), ('TSLA', '2020-08-31', 5), ('TSLA', '2022-08-25', 3),
          ('NVDA', '2021-07-20', 4), ('NVDA', '2024-06-10', 10), ('GOOGL', '2022-07-18', 20), ('AMZN', '2022-06-06', 20),
          ('NFLX', '2015-07-15', 7), ('V', '2015-03-19', 4), ('WMT', '2024-02-26', 3), ('CMG', '2024-06-26', 50),
          ('AVGO', '2024-07-15', 10), ('SHW', '2021-04-01', 3), ('ISRG', '2021-10-05', 3), ('DXCM', '2022-06-13', 4),
          ('PANW', '2022-09-14', 3), ('LRCX', '2024-10-03', 10), ('SMCI', '2024-10-01', 10), ('ORLY', '2025-06-10', 15)]


def main(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    row = {t: j for j, t in enumerate(E['tickers'])}
    L = ['ticker | split day | ratio | median DV 20 days before, $M | after, $M | after / before | close move on the day | result']
    fails = 0
    for tk, day, ratio in SPLITS:
        j = row.get(tk)
        if j is None:
            L.append(f'{tk} | {day} | {ratio} | not in export')
            fails += 1
            continue
        s = c2eval.day_index(days, day)
        dv = np.asarray(E['dollar_volume'][j, s - 20:s + 20], np.float64)
        c = np.asarray(E['close'][j, s - 1:s + 1], np.float64)
        b, a = np.nanmedian(dv[:20]), np.nanmedian(dv[20:])
        q = a / b
        jump = float(np.log(c[1] / c[0]))
        ok = abs(np.log(q)) < abs(np.log(q) - np.log(ratio)) and abs(jump) < np.log(ratio) / 2
        fails += int(not ok)
        L.append(f'{tk} | {c2eval.iso(days[s])} | {ratio} | {b / 1e6:.0f} | {a / 1e6:.0f} | {q:.2f} | {100 * jump:+.1f}% | '
                 f'{"ok" if ok else "FAIL"}')
    L.append(f'\n{len(SPLITS) - fails} of {len(SPLITS)} splits: dollar volume continuous across the split day')
    txt = '\n'.join(L)
    print(txt)
    (OUT / f'check_splits_{tag}.txt').write_text(txt + '\n')
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
