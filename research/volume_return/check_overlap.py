"""Diagnostic after the unseen run (not part of the frozen procedure): build.py's strict overlap check found many
differences between the new and old exports over the old export's days. Are they constant rescalings (later splits
or dividend re-adjustment, which leave returns and V unchanged) or changed returns?

    check_overlap.py -> out/check_overlap.txt
"""
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/volume_return')
import numpy as np  # noqa: E402

import common as C  # noqa: E402

c2eval = C.c2eval


def main():
    rep = C.OUT / 'check_overlap.txt'
    rep.write_text('')
    Eo, En = C.export('old'), C.export('new')
    To = len(Eo['days'])
    Ro = np.load(C.tagdir('old') / 'R.npy', mmap_mode='r')
    Rn = np.load(C.tagdir('new') / 'R.npy', mmap_mode='r')
    Vo = np.load(C.tagdir('old') / 'V.npy', mmap_mode='r')
    Vn = np.load(C.tagdir('new') / 'V.npy', mmap_mode='r')
    cal = dict(np.load(C.tagdir('old') / 'cal.npz'))
    rown = {t: j for j, t in enumerate(En['tickers'])}
    pairs = [(jo, rown[t]) for jo, t in enumerate(Eo['tickers']) if t in rown]
    jo = np.array([a for a, _ in pairs])
    jn = np.array([b for _, b in pairs])
    member = np.zeros(len(Eo['tickers']), bool)
    for k in range(len(cal['rebal'])):
        member |= cal['UW'][k] | cal['UJ'][k]
    C.log(rep, f'{len(pairs)} old tickers also in the new export ({member[jo].sum()} of them ever in U_W or U_J); '
               f'{len(Eo["tickers"]) - len(pairs)} old tickers absent ({int(member.sum() - member[jo].sum())} ever in U_W '
               f'or U_J)')
    n_r = n_rbad = n_v = n_vbad = 0
    bad_t_r, bad_t_v = set(), set()
    for a in range(0, len(jo), 500):
        ro = np.asarray(Ro[jo[a:a + 500]], np.float64)
        rn = np.asarray(Rn[jn[a:a + 500], :To], np.float64)
        vo = np.asarray(Vo[jo[a:a + 500]], np.float64)
        vn = np.asarray(Vn[jn[a:a + 500], :To], np.float64)
        both = np.isfinite(ro) & np.isfinite(rn)
        dr = both & (np.abs(ro - rn) > 1e-4)
        n_r += int(both.sum())
        n_rbad += int(dr.sum())
        bothv = np.isfinite(vo) & np.isfinite(vn)
        dv = bothv & (np.abs(vo - vn) > 1e-3)
        n_v += int(bothv.sum())
        n_vbad += int(dv.sum())
        for i in np.nonzero(dr.any(1))[0]:
            bad_t_r.add(int(jo[a + i]))
        for i in np.nonzero(dv.any(1))[0]:
            bad_t_v.add(int(jo[a + i]))
    C.log(rep, f'daily returns: {n_rbad} of {n_r} common finite stock-days differ by more than 1e-4 '
               f'({100 * n_rbad / max(n_r, 1):.3f}%), in {len(bad_t_r)} tickers '
               f'({sum(member[j] for j in bad_t_r)} ever in U_W or U_J)')
    C.log(rep, f'V: {n_vbad} of {n_v} common finite stock-days differ by more than 1e-3 '
               f'({100 * n_vbad / max(n_v, 1):.3f}%), in {len(bad_t_v)} tickers')
    # the same, restricted to the development and check years in U_W
    a, b = C.day_range(Eo['days'], '2008-01-01', '2025-06-27')
    tot = bad = 0
    for k in range(len(cal['rebal'])):
        t0 = int(cal['rebal'][k])
        t1 = int(cal['rebal'][k + 1]) if k + 1 < len(cal['rebal']) else To
        if t1 <= a or t0 > b:
            continue
        u = np.nonzero(cal['UW'][k][jo])[0]
        ro = np.asarray(Ro[jo[u], t0:t1], np.float64)
        rn = np.asarray(Rn[jn[u], t0:t1], np.float64)
        both = np.isfinite(ro) & np.isfinite(rn)
        tot += int(both.sum())
        bad += int((both & (np.abs(ro - rn) > 1e-4)).sum())
    C.log(rep, f'U_W member stock-days 2008..2025-06: {bad} of {tot} returns differ by more than 1e-4 '
               f'({100 * bad / max(tot, 1):.3f}%)')


if __name__ == '__main__':
    main()
