"""Overnight auctions, the reported extra (decision D11): how much of the money account's overnight profit is left if
the exit is at 9:35 instead of the opening auction. Unseen months only (new export and store), on the twin gap fade
check's Yahoo 5-minute bars (2026-07-29..2026-10-19), read-only; no bars are fetched.

Positions: the money account's 20 picks on each signal day t whose exit day d is a 5-minute window day (SPY has all
78 bars 09:30..15:55, offset -04:00). Entry = the store's close_t. Exit (a) = the store's open_d (the opening auction);
exit (b) = Yahoo's open of the 09:35 bar on d, put on the store's scale with k = store close_d / Yahoo daily close_d.
A position is in the common set if it has a valid exit by the account's rule (D6), a 09:35 bar, Yahoo daily closes on
t and d within 0.1% of the store's, and no dividend or split ex-date on d. Share kept = mean daily (b) / mean daily (a)
on the common set; also in excess of the ranked set's mean (ranked stocks with the same usable data), with 9:45 and
10:00 for context.

    python3 -I extra935.py      # -> out/extra935.md, .json
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import build as B  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

BIG = c2eval.WORK / 'twin_gap_fade_check'
TOL = 0.001


def main():
    import csv
    P = B.load('new')
    days = P['days']
    E = c2eval.load_export(C.EXPORTS['new'], arrays=('close',))
    tick = E['tickers']
    pos = np.load(BIG / 'positions.npz')
    smap = dict(zip(pos['stk_i'].tolist(), pos['stk_sym'].tolist()))
    agree = sum(1 for i, s in smap.items() if str(tick[i]).replace('.', '-') == s or str(tick[i]) == s)

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan
    o935, o945, o1000 = {}, {}, {}
    wd = set()
    with open(BIG / 'y5m.csv') as fh:
        for r in csv.DictReader(fh):
            key = (r['sym'], r['date'])
            o935[key], o945[key], o1000[key] = num(r['open_0935']), num(r['open_0945']), num(r['open_1000'])
            if (r['sym'] == 'SPY' and r['n_bars'] == '78' and r['first_time'] == '09:30' and r['last_time'] == '15:55'
                    and r['offs'] == '-04:00' and r['tz'] == 'America/New_York'):
                wd.add(r['date'])
    ycl, ev = {}, {}
    with open(BIG / 'y1d.csv') as fh:
        for r in csv.DictReader(fh):
            key = (r['sym'], r['date'])
            ycl[key] = num(r['close'])
            ev[key] = (np.nan_to_num(num(r['dividends'])) > 0) or (np.nan_to_num(num(r['stock_splits'])) > 0)
    S_ = c2eval.load_store(C.STORES['new'])
    c0 = P['c0']
    so = c2eval.store_panel(S_, E['ids'], days[c0:], 'open')
    sc = c2eval.store_panel(S_, E['ids'], days[c0:], 'close')
    ST = P['S'].T
    okT = P['ok_s'].T
    rec = []
    n_pos = n_bars = n_common = 0
    drop = {'no_bar': 0, 'close_disagrees': 0, 'event': 0, 'exit_invalid': 0}
    for d in range(c2eval.day_index(days, '2026-07-01'), len(days)):
        ds = iso(days[d])
        if ds not in wd:
            continue
        t = d - 1
        k = C.kidx(P['rebal'], t - 1)
        U = P['uni'][k] & (P['fund_row'][k] >= 0)
        sv = np.asarray(ST[t], np.float64)
        rows = np.nonzero(U & np.isfinite(sv))[0]
        top = set(rows[np.argsort(-sv[rows], kind='stable')][:C.PICKS].tolist())
        tsd = iso(days[t])
        okd = np.asarray(okT[d - c0])
        for i in rows:
            sym = smap.get(int(i), str(tick[i]).replace('.', '-'))
            ins = i in top
            if ins:
                n_pos += 1
            if not okd[i]:
                if ins:
                    drop['exit_invalid'] += 1
                continue
            p935 = o935.get((sym, ds), np.nan)
            if not np.isfinite(p935):
                if ins:
                    drop['no_bar'] += 1
                continue
            if ins:
                n_bars += 1
            ct, cd = float(sc[i, t - c0]), float(sc[i, d - c0])
            yt, yd_ = ycl.get((sym, tsd), np.nan), ycl.get((sym, ds), np.nan)
            if not (np.isfinite(yt) and np.isfinite(yd_) and abs(ct / yt - 1) <= TOL and abs(cd / yd_ - 1) <= TOL):
                if ins:
                    drop['close_disagrees'] += 1
                continue
            if ev.get((sym, ds), False):
                if ins:
                    drop['event'] += 1
                continue
            kk = cd / yd_
            r_a = float(so[i, d - c0]) / ct - 1
            rr = {'d': d, 'i': int(i), 'top': ins, 'a': r_a}
            for nm, dct in (('935', o935), ('945', o945), ('1000', o1000)):
                p = dct.get((sym, ds), np.nan)
                rr[nm] = p * kk / ct - 1 if np.isfinite(p) else np.nan
            rec.append(rr)
            if ins:
                n_common += 1
    cols = ('a', '935', '945', '1000')
    rec = [r for r in rec if all(np.isfinite(r[c]) for c in cols)]
    dd_ = np.array([r['d'] for r in rec])
    tp = np.array([r['top'] for r in rec])
    V = np.array([[r[c] for c in cols] for r in rec])
    ud = np.unique(dd_[tp])
    acc = {c: np.array([V[(dd_ == d) & tp, j].mean() for d in ud]) for j, c in enumerate(cols)}
    uni = {c: np.array([V[dd_ == d, j].mean() for d in ud]) for j, c in enumerate(cols)}
    res = {'window_days': len(ud), 'first': iso(days[int(ud.min())]), 'last': iso(days[int(ud.max())]),
           'positions': n_pos, 'with_935_bar': n_bars, 'common': int(tp.sum()), 'ranked_rows': int(len(rec)), 'drops': drop,
           'symbol_map_agree_with_tickers': agree, 'symbol_map_size': len(smap)}
    out = {}
    for nm in ('a', '935', '945', '1000'):
        m, t_, _ = C.tstat(acc[nm])
        x = acc[nm] - uni[nm]
        mx, tx, _ = C.tstat(x)
        out[nm] = {'account_bps': m * 1e4, 't': t_, 'ranked_bps': float(uni[nm].mean()) * 1e4, 'excess_bps': mx * 1e4, 'excess_t': tx}
    for nm in ('935', '945', '1000'):
        out[nm]['share_kept'] = out[nm]['account_bps'] / out['a']['account_bps']
        out[nm]['share_kept_excess'] = out[nm]['excess_bps'] / out['a']['excess_bps']
        dd = acc[nm] - acc['a']
        m, t_, _ = C.tstat(dd)
        out[nm]['minus_open_bps'] = m * 1e4
        out[nm]['minus_open_t'] = t_
    res['exits'] = out
    (C.OUT / 'extra935.json').write_text(json.dumps(res, indent=1))
    L = [f'# Overnight auctions: exit at 9:35 instead of the open (unseen months, Yahoo 5-minute bars)\n',
         f"Window exit days {res['first']}..{res['last']} ({res['window_days']}); account positions {n_pos}, with a 09:35 bar "
         f"{n_bars}, in the common set {res['common']} ({res['common'] / max(n_pos, 1):.0%}); drops {drop}. Symbol map: "
         f"{agree} of {len(smap)} Yahoo symbols equal the export ticker (dots as dashes).\n",
         '| Exit | Account, bps a day (t) | Ranked set, bps a day | Account minus ranked (t) | Share of the open exit kept | Share kept, excess | Minus the open exit (t) |',
         '|---|---|---|---|---|---|---|']
    for nm, lab in (('a', 'opening auction (store open)'), ('935', '09:35'), ('945', '09:45'), ('1000', '10:00')):
        o = out[nm]
        L.append(f"| {lab} | {o['account_bps']:+.1f} ({o['t']:+.2f}) | {o['ranked_bps']:+.1f} | {o['excess_bps']:+.1f} ({o['excess_t']:+.2f}) | "
                 f"{o.get('share_kept', 1.0):.2f} | {o.get('share_kept_excess', 1.0):.2f} | "
                 + (f"{o['minus_open_bps']:+.1f} ({o['minus_open_t']:+.2f})" if nm != 'a' else '') + ' |')
    txt = '\n'.join(L) + '\n'
    (C.OUT / 'extra935.md').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main()
