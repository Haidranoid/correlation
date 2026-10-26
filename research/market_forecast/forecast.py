"""P1, P2 and checks C1, C2, C3, C5 and C7, on the development and check periods only (targets ending by 2025-06).
P1' and P2' repeat P1 and P2 against the two-lag benchmark (Decisions 2), and every development result of P1, P1',
P2 and P2' is placed among C9's noise sets of the same size and benchmark (Decisions 3). The unseen months are
scored once, at the end, by unseen.py from the forecasts saved here.

    python3 -I -B forecast.py     # -> forecast.txt, forecast.json, WORK/forecasts.npz
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_common as M  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = {k: i for i, k in enumerate(M.INPUTS)}
PERIODS = (('dev', M.DEV), ('check', M.CHECK))


def run(X, series, h, lag=0, two=False):
    out = {}
    for b in M.MARKETS:
        months, y, g, Xm = M.panel(X, series[b], M.bls(M.PRICE[b]), h, lag=lag)
        out[b] = {'y': y, 'g': g, **M.oos(y, M.two_lag(g) if two else g, Xm, months, h)}
    return months, out


NOISE = json.loads((HERE / 'codecheck.json').read_text())['noise']


def vs_noise(e, bench, size):
    """Shares of C9's noise sets (same benchmark and number of series) below this result's R^2 and t, and with a
    bootstrap p at or above its p."""
    q = NOISE[bench][str(size)]
    return {'r2_pct': float(np.mean(np.array(q['r2']) < e['r2os'])),
            't_pct': float(np.mean(np.array(q['t']) < e['cw_t'])),
            'p_pct': float(np.mean(np.array(q['boot_p']) >= e.get('cw_boot_p', 1.0))),
            'noise_r2_95': q['pct_r2']['95']}


def fmt_noise(v):
    return (f'among noise: R2 above {100 * v["r2_pct"]:.1f}% (noise 95th {v["noise_r2_95"]:+.4f}), t above '
            f'{100 * v["t_pct"]:.1f}%, bootstrap p below {100 * v["p_pct"]:.1f}%')


def pooled(R, months, h, key, mask_extra=None, markets=tuple(M.MARKETS), base='fb', boot=True):
    """{period: evaluate} for forecast `key` (a column list of fk, or 'fs'/'fb') against `base`."""
    res = {}
    for p, rng in PERIODS:
        mask = M.period_mask(months, h, *rng)
        if mask_extra is not None:
            mask &= mask_extra
        fm = [R[b][key] if isinstance(key, str) else M.combo(R[b]['fk'], key) for b in markets]
        res[p] = M.evaluate([R[b]['y'] for b in markets], [R[b][base] for b in markets], fm, mask, boot=boot)
    return res


def passes(e, p_bar=0.0013, p=None):
    p = e.get('cw_boot_p', 1.0) if p is None else p
    return e['cw_t'] >= 3 and p < p_bar and e['r2os'] > 0


def consistent(e):
    return e.get('n', 0) >= 12 and e['r2os'] > 0 and e['cw_p'] < 0.05


ALL = [KS[k] for k in M.INPUTS]


def main():
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    X = M.common_inputs()
    qty = {b: M.bea(c) for b, c in M.MARKETS.items()}
    months, R12 = run(X, qty, 12)
    _, R3 = run(X, qty, 3)
    _, T12 = run(X, qty, 12, two=True)
    np.savez(M.WORK / 'forecasts.npz', months=months,
             **{f'{k}_{b}_{nm}': R[b][k] for nm, R in (('h12', R12), ('h3', R3), ('h12two', T12)) for b in M.MARKETS
                for k in ('y', 'g', 'f0', 'fb', 'fk', 'fs')})

    # P1
    out('P1: the model (nine inputs) against own history, next 12 months, spending adjusted for prices, '
        'wine + beer + spirits + restaurants pooled')
    e = pooled(R12, months, 12, ALL)
    res['P1'] = e
    for p in ('dev', 'check'):
        out(f'  {p}: {M.fmt_ev(e[p])}')
    res['P1_noise'] = vs_noise(e['dev'], 'one-lag', 9)
    out(f'  dev {fmt_noise(res["P1_noise"])}')
    res['P1_pass'] = bool(passes(e['dev']))
    res['P1_check_consistent'] = bool(consistent(e['check']))
    out(f'  pass (t >= 3, bootstrap p < 0.0013, R2 > 0): {res["P1_pass"]}; check consistent (R2 > 0, normal p < 0.05): '
        f'{res["P1_check_consistent"]}')
    res['P1_market'] = {}
    for b in M.MARKETS:
        eb = pooled(R12, months, 12, ALL, markets=(b,))
        res['P1_market'][b] = eb
        out(f'  {M.NAMES[b]}: dev {M.fmt_ev(eb["dev"])}')
        out(f'  {" " * len(M.NAMES[b])}  check {M.fmt_ev(eb["check"])}')
    # by year of the target's end, pooled R2
    end_year = (months + 12) // 12
    byy = {}
    for yr in range(2009, 2026):
        msk = (end_year == yr) & M.period_mask(months, 12, M.DEV[0], M.CHECK[1])
        ev = M.evaluate([R12[b]['y'] for b in M.MARKETS], [R12[b]['fb'] for b in M.MARKETS],
                        [M.combo(R12[b]['fk'], ALL) for b in M.MARKETS], msk, boot=False)
        if ev.get('n', 0) >= 12:
            byy[yr] = {'r2os': ev['r2os'], 'mae_b': ev['mae_b'], 'mae_m': ev['mae_m']}
    res['P1_by_year'] = byy
    out('  by year of target end, pooled R2: ' + ', '.join(f'{y} {v["r2os"]:+.2f}' for y, v in byy.items()))
    # without targets that touch 2020 (the window L-11..L+h)
    touch = np.array([any(M.mkey(m)[:4] == '2020' for m in range(L - 11, L + 13)) for L in months])
    e20 = pooled(R12, months, 12, ALL, mask_extra=~touch)
    res['P1_no2020'] = e20
    out(f'  check without targets whose windows touch 2020: {M.fmt_ev(e20["check"])}')

    # P1': the two-lag benchmark
    out()
    out("P1': the same against the two-lag benchmark (own growth now and a year earlier; Decisions 2)")
    e2 = pooled(T12, months, 12, ALL)
    res["P1'"] = e2
    for p in ('dev', 'check'):
        out(f'  {p}: {M.fmt_ev(e2[p])}')
    res["P1'_noise"] = vs_noise(e2['dev'], 'two-lag', 9)
    out(f'  dev {fmt_noise(res["P1\'_noise"])}')
    res["P1'_market"] = {b: pooled(T12, months, 12, ALL, markets=(b,)) for b in M.MARKETS}
    for b, eb in res["P1'_market"].items():
        out(f'  {M.NAMES[b]}: dev R2 {eb["dev"]["r2os"]:+.4f} t {eb["dev"]["cw_t"]:+.2f}; check R2 '
            f'{eb["check"]["r2os"]:+.4f} t {eb["check"]["cw_t"]:+.2f}')
    res['bench_two_vs_one'] = {p: M.evaluate([R12[b]['y'] for b in M.MARKETS], [R12[b]['fb'] for b in M.MARKETS],
                                             [T12[b]['fb'] for b in M.MARKETS], M.period_mask(months, 12, *rng))
                               for p, rng in PERIODS}
    for p in ('dev', 'check'):
        out(f'  the two-lag benchmark against the one-lag one, {p}: {M.fmt_ev(res["bench_two_vs_one"][p])}')

    # P2
    out()
    out('P2: each thesis alone against own history (same pooled test); Holm across the five on the bootstrap p')
    res['P2'] = {}
    for th, ks in M.THESES.items():
        res['P2'][th] = pooled(R12, months, 12, [KS[k] for k in ks])
    hp = M.holm([res['P2'][th]['dev'].get('cw_boot_p', 1.0) for th in M.THESES])
    for (th, ev), h_ in zip(res['P2'].items(), hp):
        ev['dev']['holm_p'] = float(h_)
        ev['pass'] = bool(passes(ev['dev'], p=h_))
        ev['check_consistent'] = bool(consistent(ev['check']))
        ev['noise'] = vs_noise(ev['dev'], 'one-lag', len(M.THESES[th]))
        out(f'  {th}: dev {M.fmt_ev(ev["dev"])}; Holm p {h_:.4f}; pass {ev["pass"]}')
        out(f'  {" " * len(th)}  dev {fmt_noise(ev["noise"])}')
        out(f'  {" " * len(th)}  check {M.fmt_ev(ev["check"])}; consistent {ev["check_consistent"]}')
    out("P2': each thesis against the two-lag benchmark")
    res["P2'"] = {}
    for th, ks in M.THESES.items():
        ev = pooled(T12, months, 12, [KS[k] for k in ks])
        ev['noise'] = vs_noise(ev['dev'], 'two-lag', len(ks))
        res["P2'"][th] = ev
        out(f'  {th}: dev R2 {ev["dev"]["r2os"]:+.4f} t {ev["dev"]["cw_t"]:+.2f} boot p {ev["dev"]["cw_boot_p"]:.4f} '
            f'({fmt_noise(ev["noise"])}); check R2 {ev["check"]["r2os"]:+.4f} t {ev["check"]["cw_t"]:+.2f}')

    # C1
    out()
    out('C1: each input alone (pooled; Holm across nine for information), the kitchen sink, and both against the mean')
    res['C1'] = {'inputs': {}, 'wine': {}}
    for k in M.INPUTS:
        res['C1']['inputs'][k] = pooled(R12, months, 12, [KS[k]])
        res['C1']['wine'][k] = pooled(R12, months, 12, [KS[k]], markets=('W',))
    hp = M.holm([res['C1']['inputs'][k]['dev'].get('cw_boot_p', 1.0) for k in M.INPUTS])
    for k, h_ in zip(M.INPUTS, hp):
        ev, ew = res['C1']['inputs'][k], res['C1']['wine'][k]
        ev['dev']['holm_p'] = float(h_)
        out(f'  {k}: pooled dev R2 {ev["dev"]["r2os"]:+.4f} t {ev["dev"]["cw_t"]:+.2f} (Holm p {h_:.3f}); '
            f'check R2 {ev["check"]["r2os"]:+.4f} t {ev["check"]["cw_t"]:+.2f} | wine dev R2 {ew["dev"]["r2os"]:+.4f} '
            f't {ew["dev"]["cw_t"]:+.2f}, check R2 {ew["check"]["r2os"]:+.4f} t {ew["check"]["cw_t"]:+.2f}')
    res['C1']['sink'] = pooled(R12, months, 12, 'fs')
    out(f'  kitchen sink: dev {M.fmt_ev(res["C1"]["sink"]["dev"])}')
    out(f'                check {M.fmt_ev(res["C1"]["sink"]["check"])}')
    res['C1']['model_vs_mean'] = pooled(R12, months, 12, ALL, base='f0')
    res['C1']['bench_vs_mean'] = pooled(R12, months, 12, 'fb', base='f0')
    for nm in ('bench_vs_mean', 'model_vs_mean'):
        for p in ('dev', 'check'):
            out(f'  {nm.replace("_", " ")}, {p}: {M.fmt_ev(res["C1"][nm][p])}')

    # C2: three months ahead
    out()
    out('C2: three months ahead (the next quarter against the same quarter a year earlier)')
    res['C2'] = {'P1': pooled(R3, months, 3, ALL), 'P2': {}}
    for p in ('dev', 'check'):
        out(f'  P1 {p}: {M.fmt_ev(res["C2"]["P1"][p])}')
    for th, ks in M.THESES.items():
        res['C2']['P2'][th] = pooled(R3, months, 3, [KS[k] for k in ks])
        out(f'  {th}: dev R2 {res["C2"]["P2"][th]["dev"]["r2os"]:+.4f} t {res["C2"]["P2"][th]["dev"]["cw_t"]:+.2f}; '
            f'check R2 {res["C2"]["P2"][th]["check"]["r2os"]:+.4f} t {res["C2"]["P2"][th]["check"]["cw_t"]:+.2f}')
    res['C2']['bench_vs_mean'] = pooled(R3, months, 3, 'fb', base='f0')
    out(f'  own history against the mean: dev R2 {res["C2"]["bench_vs_mean"]["dev"]["r2os"]:+.4f}, '
        f'check R2 {res["C2"]["bench_vs_mean"]["check"]["r2os"]:+.4f}')

    # C3: dollars
    out()
    _, RD = run(X, {b: M.bea(c) for b, c in M.DOLLARS.items()}, 12)
    res['C3'] = pooled(RD, months, 12, ALL)
    for p in ('dev', 'check'):
        out(f'C3: current dollars, {p}: {M.fmt_ev(res["C3"][p])}')

    # C5: errors in plain units
    out()
    out('C5: mean absolute error of next-12-month growth, percentage points (direction right: share of origins where '
        'the forecast and the outcome were on the same side of last year\'s growth)')
    res['C5'] = {}
    for p, rng in PERIODS:
        mask = M.period_mask(months, 12, *rng)
        res['C5'][p] = {}
        for b in M.MARKETS:
            r, r2 = R12[b], T12[b]
            fm, fm2 = M.combo(r['fk'], ALL), M.combo(r2['fk'], ALL)
            ok = mask & np.isfinite(r['y']) & np.isfinite(fm) & np.isfinite(r['fb']) & np.isfinite(fm2)
            y, g = r['y'][ok], r['g'][ok]
            row = {}
            for nm, f in (('mean', r['f0'][ok]), ('no change', g), ('own history', r['fb'][ok]), ('model', fm[ok]),
                          ('own history, 2 lags', r2['fb'][ok]), ('model, 2 lags', fm2[ok])):
                row[nm] = {'mae': float(100 * np.mean(np.abs(y - f))),
                           'direction': None if nm == 'no change' else float(np.mean(np.sign(f - g) == np.sign(y - g)))}
            res['C5'][p][b] = row
            out(f'  {p} {M.NAMES[b]} (n {int(ok.sum())}): ' + '; '.join(
                f'{nm} {v["mae"]:.2f}' + ('' if v['direction'] is None else f' ({100 * v["direction"]:.0f}% dir)')
                for nm, v in row.items()))

    # C7: every input a month older
    out()
    _, R7 = run(X, qty, 12, lag=1)
    res['C7'] = pooled(R7, months, 12, ALL)
    for p in ('dev', 'check'):
        out(f'C7: inputs one month older, {p}: {M.fmt_ev(res["C7"][p])}')

    (HERE / 'forecast.txt').write_text('\n'.join(lines) + '\n')
    M.dump(res, HERE / 'forecast.json')


if __name__ == '__main__':
    main()
