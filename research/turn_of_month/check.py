"""Plumbing checks, before any outcome is computed: calendars, Yahoo SPY's dates against the exports', NaNs, TOM day
counts per month. Writes check.txt. No return statistic by TOM label is computed here."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, pubdata  # noqa: E402

out = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    out.append(s)


Eo = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('market',))
En = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('market',))
do, dn = Eo['days'], En['days']
say('old export days', len(do), c2eval.iso(do[0]), '..', c2eval.iso(do[-1]))
say('new export days', len(dn), c2eval.iso(dn[0]), '..', c2eval.iso(dn[-1]))
say('new export first len(old) days equal old:', bool(np.array_equal(dn[:len(do)], do)))

yd, px = pubdata.etf_adjclose(['SPY'])
yday = pubdata.to_days(yd[np.isfinite(px['SPY'])])
for name, days, lo, hi in (('old span', do, '2005-12-01', '2025-06-27'), ('unseen', dn, '2025-06-27', '2026-10-19')):
    m = C.span_mask(days, lo, hi)
    ed = days[m]
    ym = (yday >= C.dnum(lo)) & (yday <= C.dnum(hi))
    yy = yday[ym]
    only_e = np.setdiff1d(ed, yy)
    only_y = np.setdiff1d(yy, ed)
    say(f'{name}: export days {len(ed)}, Yahoo SPY days {len(yy)}, only in export {len(only_e)} '
        f'{[c2eval.iso(x) for x in only_e[:10]]}, only in Yahoo {len(only_y)} {[c2eval.iso(x) for x in only_y[:10]]}')

r_o, P_o, _ = C.spy_total(do)
r_n, P_n, _ = C.spy_total(dn)
m = C.span_mask(do, *C.SPAN)
say('old span: SPY return NaN days', int(np.sum(~np.isfinite(r_o[m]))), '; first SPY price day in export',
    c2eval.iso(do[np.argmax(np.isfinite(P_o))]))
mu = C.span_mask(dn, *C.UNSEEN)
say('unseen: SPY return NaN days', int(np.sum(~np.isfinite(r_n[mu]))), 'days', int(mu.sum()))
say('old/new SPY return identical over old span:', bool(np.allclose(r_o[m], r_n[:len(do)][m], rtol=0, atol=1e-12)))
say('SPY |return| > 7% days in span:', [(c2eval.iso(d), round(float(x), 4)) for d, x in zip(do[m], r_o[m]) if abs(x) > 0.07])

rf_o = pubdata.rf_daily(do)
rf_n = pubdata.rf_daily(dn)
say('rf NaN in span', int(np.sum(~np.isfinite(rf_o[m]))), 'unseen', int(np.sum(~np.isfinite(rf_n[mu]))))
say('rf annualized mean over span (x252)', round(float(np.mean(rf_o[m]) * 252), 4))
mk_o = np.asarray(Eo['market'], float)
mk_n = np.asarray(En['market'], float)
say('market NaN in span', int(np.sum(~np.isfinite(mk_o[m]))), 'unseen', int(np.sum(~np.isfinite(mk_n[mu]))))
say('old/new market max abs diff over old span:', float(np.nanmax(np.abs(mk_o[m] - mk_n[:len(do)][m]))))

# TOM labels: walk-forward on the old export's days, month ends from the new export's calendar
lab_o = C.tom_label(do, dn)
lab_o_self = C.tom_label(do, do)
diff = np.nonzero(lab_o != lab_o_self)[0]
say('labels differ (old-only calendar vs new calendar) on', [(c2eval.iso(do[i]), int(lab_o_self[i]), int(lab_o[i])) for i in diff])
y, mo = C.ymd(do[m])
lab = lab_o[m]
keys = y * 12 + mo
cnt = {}
for k in np.unique(keys):
    s = lab[keys == k]
    cnt[k] = (int((s != 0).sum()), int(len(s)))
ndays = np.array([v[1] for v in cnt.values()])
nt = np.array([v[0] for v in cnt.values()])
say('span months', len(cnt), '; TOM days per month: values', dict(zip(*np.unique(nt, return_counts=True))),
    '; trading days per month min/max', ndays.min(), ndays.max())
for P_, (lo, hi) in C.PERIODS.items():
    mm = C.span_mask(do, lo, hi)
    say(f'{P_}: days {int(mm.sum())}, TOM {int((lab_o[mm] != 0).sum())}, by position',
        {p: int((lab_o[mm] == p).sum()) for p in C.POSITIONS})
lab_n = C.tom_label(dn[mu], dn)
say('unseen: days', int(mu.sum()), 'TOM', int((lab_n != 0).sum()), {p: int((lab_n == p).sum()) for p in C.POSITIONS},
    'first day', c2eval.iso(dn[mu][0]), 'label', int(lab_n[0]), 'last', c2eval.iso(dn[mu][-1]))
say('old export labels equal new export labels on old days:', bool(np.array_equal(lab_o, C.tom_label(dn, dn)[:len(do)])))
(C.HERE / 'check.txt').write_text('\n'.join(out) + '\n')
