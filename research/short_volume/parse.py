"""Short volume: parse the FINRA daily files into [N, T] panels aligned to an export.

Rules (experiments/+0.0-short-volume.md, "Decisions made while building it"):
  - Per symbol and day, ShortVolume and TotalVolume are summed over the TRF and ADF facility files, FNSQ + FNQC + FNYX
    + FNRA: the set FINRA's consolidated CNMS file combines. A facility file that doesn't exist (403) counts as no
    volume there. The ORF (FORF, OTC securities) is left out.
  - If the day lacks FNSQ or FNYX (or FNQC from 2018-09-10) but has CNMS, CNMS is used instead; if neither, the day
    is missing.
  - Symbols: '.', '/' and ' ' become '-', then exact match to the export's ticker (BRK.B, BRK/B -> BRK-B).
  - Checks: the facility sum against CNMS on every day both exist; the Date field against the file name; the monthly
    FORF samples against the export's tickers.

    python3 -I -B parse.py old|new
Writes WORK/short_volume/<tag>/{short,total,cn_short,cn_total}.npy ([N, T] float32, NaN = no record) and
day_info.npz, plus out/parse_<tag>.txt and out/parse_<tag>.json.
"""
import sys
sys.dont_write_bytecode = True
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

WORKD = c2eval.WORK / 'short_volume'
RAW = WORKD / 'raw'
MANIFEST = WORKD / 'manifest.tsv'
EXPORTS = {'old': c2eval.EXPORT_OLD, 'new': c2eval.EXPORT_NEW}
FIRST = '2009-08-03'
FAC = ('FNSQ', 'FNQC', 'FNYX', 'FNRA')
T0 = time.time()
NOHEAD = []   # files without a header row
FRAC = [None]  # first file (in date order) with a fractional volume


def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def norm(sym):
    return sym.strip().replace('.', '-').replace('/', '-').replace(' ', '-')


def read_file(path, ymd):
    """{symbol: [short, total]} (ints) from one FINRA file; also the count of Date mismatches and the trailer check."""
    out = {}
    bad_date = 0
    lines = path.read_text(errors='replace').splitlines()
    if not lines:
        return out, 0, None
    head = lines[0].split('|')
    if 'ShortVolume' in head:
        i_s, i_t = head.index('ShortVolume'), head.index('TotalVolume')
        nf, body = len(head), lines[1:]
    else:  # a few files have no header row: the layout follows from the field count
        nf = len(head)
        i_s, i_t = {5: (2, 3), 6: (2, 4)}[nf]
        body = lines
        NOHEAD.append(path.name)
    n_rec = 0
    trailer = None
    for l in body:
        f = l.split('|')
        if len(f) < nf:
            if l.strip().isdigit():
                trailer = int(l.strip())
            continue
        n_rec += 1
        if f[0] != ymd:
            bad_date += 1
        if FRAC[0] is None and ('.' in f[i_s] or '.' in f[i_t]):
            FRAC[0] = path.name
        try:
            s, t = float(f[i_s]), float(f[i_t])   # fractional share volumes appear in later files
        except ValueError:
            continue
        sym = f[1]
        if sym in out:  # duplicate symbol within one file: add
            out[sym][0] += s
            out[sym][1] += t
        else:
            out[sym] = [s, t]
    return out, bad_date, (trailer == n_rec) if trailer is not None else None


def main(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=())
    days = E['days']
    tickers = E['tickers']
    N, T = len(tickers), len(days)
    row = {}
    dup = 0
    for j, t in enumerate(tickers):
        k = norm(str(t))
        if k in row:
            dup += 1
        row[k] = j
    log(tag, 'N', N, 'T', T, 'duplicate normalized tickers', dup)
    man = {}
    for l in MANIFEST.read_text().splitlines()[1:]:
        f = l.split('\t')
        man[f[2]] = int(f[3])
    iso = [c2eval.iso(d) for d in days]
    d0 = iso.index(FIRST)
    short = np.full((N, T), np.nan, np.float32)
    total = np.full((N, T), np.nan, np.float32)
    cn_s = np.full((N, T), np.nan, np.float32)
    cn_t = np.full((N, T), np.nan, np.float32)
    src = np.zeros(T, np.int8)          # 0 none, 1 facility sum, 2 CNMS fallback
    have = np.zeros((T, 6), bool)       # FNSQ FNQC FNYX FNRA CNMS FORF
    PRE = ('FNSQ', 'FNQC', 'FNYX', 'FNRA', 'CNMS', 'FORF')
    nsym = np.zeros(T, np.int32)
    nmatch = np.zeros(T, np.int32)
    bad_dates = 0
    trailer_bad = 0
    punct = defaultdict(lambda: defaultdict(int))   # year -> separator -> count of symbols with it
    unmatched_vol = defaultdict(lambda: defaultdict(int))  # year -> symbol -> total volume
    check = []    # per CNMS day: date, n_fac, n_cn, n_both, n_exact, sum_t_fac, sum_t_cn, sum_s_fac, sum_s_cn
    orf_hits = []  # (date, ticker, orf_total, fac_total)
    missing_days = []
    for d in range(d0, T):
        ymd = iso[d].replace('-', '')
        files = {p: RAW / f'{p}shvol{ymd}.txt' for p in PRE}
        for i, p in enumerate(PRE):
            have[d, i] = man.get(files[p].name) == 200 and files[p].exists()
        fac = {}
        for i, p in enumerate(FAC):
            if have[d, PRE.index(p)]:
                rec, bd, tr = read_file(files[p], ymd)
                bad_dates += bd
                trailer_bad += int(tr is False)
                for sym, (s, t) in rec.items():
                    if sym in fac:
                        fac[sym][0] += s
                        fac[sym][1] += t
                    else:
                        fac[sym] = [s, t]
        cn = None
        if have[d, PRE.index('CNMS')]:
            cn, bd, tr = read_file(files['CNMS'], ymd)
            bad_dates += bd
            trailer_bad += int(tr is False)
        need = ['FNSQ', 'FNYX'] + (['FNQC'] if iso[d] >= '2018-09-10' else [])
        fac_ok = all(have[d, PRE.index(p)] for p in need)
        if fac_ok:
            use, src[d] = fac, 1
        elif cn is not None:
            use, src[d] = cn, 2
        else:
            missing_days.append(iso[d])
            continue
        y = iso[d][:4]
        nsym[d] = len(use)
        for sym, (s, t) in use.items():
            for sep in ('.', '/', ' ', '-'):
                if sep in sym:
                    punct[y][sep] += 1
            j = row.get(norm(sym))
            if j is None:
                unmatched_vol[y][sym] += t
                continue
            nmatch[d] += 1
            short[j, d] = s
            total[j, d] = t
        if cn is not None:
            for sym, (s, t) in cn.items():
                j = row.get(norm(sym))
                if j is not None:
                    cn_s[j, d] = s
                    cn_t[j, d] = t
            if fac_ok:
                both = set(fac) & set(cn)
                exact = sum(1 for k in both if abs(fac[k][0] - cn[k][0]) < 0.01 and abs(fac[k][1] - cn[k][1]) < 0.01)
                check.append((iso[d], len(fac), len(cn), len(both), exact,
                              sum(v[1] for v in fac.values()), sum(v[1] for v in cn.values()),
                              sum(v[0] for v in fac.values()), sum(v[0] for v in cn.values())))
        if have[d, PRE.index('FORF')]:
            orf, _, _ = read_file(files['FORF'], ymd)
            for sym, (s, t) in orf.items():
                j = row.get(norm(sym))
                if j is not None:
                    orf_hits.append((iso[d], str(tickers[j]), int(j), t, fac.get(sym, [0, 0])[1]))
        if d % 250 == 0:
            log(iso[d], 'symbols', nsym[d], 'matched', nmatch[d])
    out = WORKD / tag
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'short.npy', short)
    np.save(out / 'total.npy', total)
    np.save(out / 'cn_short.npy', cn_s)
    np.save(out / 'cn_total.npy', cn_t)
    np.savez(out / 'day_info.npz', src=src, have=have, nsym=nsym, nmatch=nmatch, d0=d0)
    # report
    lines = [f'export {tag}: N {N}, T {T}; first FINRA day {FIRST} (export index {d0}); duplicate normalized tickers {dup}']
    lines.append(f'days from {FIRST}: {T - d0}; facility sum used {int((src == 1).sum())}, CNMS fallback {int((src == 2).sum())}, '
                 f'missing {len(missing_days)}: {missing_days[:20]}')
    lines.append(f'Date-field mismatches {bad_dates}; files whose trailer count != records {trailer_bad}; '
                 f'files without a header row {len(NOHEAD)} (e.g. {NOHEAD[:5]}); first file with fractional volumes {FRAC[0]}')
    lines.append('files present by prefix and year (days with the file / days):')
    years = sorted(set(i[:4] for i in iso[d0:]))
    for y in years:
        m = np.array([iso[d][:4] == y for d in range(T)])
        m[:d0] = False
        lines.append(f'  {y}: days {int(m.sum())}; ' + ', '.join(f'{p} {int(have[m, i].sum())}' for i, p in enumerate(PRE)))
    lines.append('symbol separators by year (count of symbol-days with ".", "/", " ", "-"):')
    for y in years:
        lines.append(f'  {y}: ' + ', '.join(f'"{k}" {v}' for k, v in sorted(punct[y].items())))
    lines.append('FINRA symbols matched to an export ticker, by year (symbol-days):')
    for y in years:
        m = np.array([iso[d][:4] == y for d in range(T)])
        m[:d0] = False
        lines.append(f'  {y}: {int(nmatch[m].sum())} of {int(nsym[m].sum())} ({nmatch[m].sum() / max(1, nsym[m].sum()):.1%})')
    lines.append('largest unmatched symbols by total volume, by year (top 15):')
    for y in years:
        top = sorted(unmatched_vol[y].items(), key=lambda kv: -kv[1])[:15]
        lines.append(f'  {y}: ' + ', '.join(f'{k}' for k, v in top))
    if check:
        c = np.array([x[1:] for x in check], np.float64)
        lines.append(f'facility sum vs CNMS on {len(check)} days ({check[0][0]}..{check[-1][0]}):')
        lines.append(f'  symbols: facility mean {c[:, 0].mean():.0f}, CNMS {c[:, 1].mean():.0f}, both {c[:, 2].mean():.0f}; '
                     f'identical to 0.01 shares (short and total) {c[:, 3].sum() / c[:, 2].sum():.4%} of symbols in both')
        lines.append(f'  total volume: facility / CNMS = {c[:, 4].sum() / c[:, 5].sum():.6f}; short: {c[:, 6].sum() / c[:, 7].sum():.6f}')
        worst = sorted(check, key=lambda x: x[4] / max(1, x[3]))[:5]
        lines.append('  days with the lowest identical share: ' + '; '.join(f'{x[0]} {x[4]}/{x[3]}' for x in worst))
    lines.append(f'ORF (FORF) monthly samples: {int(have[:, 5].sum())} files; records matching an export ticker: {len(orf_hits)}')
    if orf_hits:
        lines.append('  e.g. ' + '; '.join(f'{h[0]} {h[1]} orf {h[3]} fac {h[4]}' for h in orf_hits[:15]))
    txt = '\n'.join(lines) + '\n'
    (HERE / 'out' / f'parse_{tag}.txt').write_text(txt)
    json.dump({'orf_hits': orf_hits, 'check': check, 'missing_days': missing_days},
              open(HERE / 'out' / f'parse_{tag}.json', 'w'))
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1])
