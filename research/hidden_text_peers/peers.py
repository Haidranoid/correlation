"""Text peers per fiscal year, Hoberg & Phillips style.

For each fiscal year:
- one filing per CIK: the one with the most Item 1 words; kept when Item 1 has at least 1,000 words;
- words in more than 25% of those filings, or in fewer than 3, are dropped;
- binary word vectors, L2-normalized: cosine(i, j) = |words_i & words_j| / sqrt(|words_i| |words_j|);
- the cutoff is the 98th percentile of the similarity over all pairs i < j of the year's filings (the top 2%);
- for every filing of an export-mapped CIK, every other filing at or above the cutoff is stored with its rank among
  all filings and among mapped filings, so the 50-peer cap can be applied either way later. The matrix is discarded.

    python3 -I peers.py                 # reads text/{fy}.npz, writes peers/{fy}.npz and peers_counts.txt here
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (CAP, FY_FIRST, FY_LAST, HERE, MAX_DF, MIN_DF, MIN_ITEM1_WORDS, PEERS, TEXT, TOP_PAIRS,  # noqa: E402
                    avail_day, cik_rows, log, np)

COL_BLOCK = 8192


def one_year(fy, mapped_ciks):
    z = np.load(TEXT / f'{fy}.npz')
    cik, fye, ntok, ptr, ids = z['cik'], z['fye'], z['ntok'], z['ptr'], z['ids']
    n_all = len(cik)
    # one filing per CIK: the most Item 1 words (first on ties)
    best = {}
    for i in range(n_all):
        c = int(cik[i])
        if c not in best or ntok[i] > ntok[best[c]]:
            best[c] = i
    sel = np.array(sorted(i for i in best.values() if ntok[i] >= MIN_ITEM1_WORDS), np.int64)
    n = len(sel)
    lens = ptr[sel + 1] - ptr[sel]
    rows = np.repeat(np.arange(n), lens)
    cols = np.concatenate([ids[ptr[i]:ptr[i + 1]] for i in sel])
    df = np.bincount(cols, minlength=int(ids.max()) + 1)
    keep = (df >= MIN_DF) & (df <= MAX_DF * n)
    newcol = np.full(len(df), -1, np.int64)
    newcol[keep] = np.arange(int(keep.sum()))
    V = int(keep.sum())
    m = keep[cols]
    rows, cols = rows[m], newcol[cols[m]]
    k = np.bincount(rows, minlength=n).astype(np.float64)       # kept words per filing
    assert (k > 0).all(), 'a filing has no kept words'
    # intersection counts, by column blocks
    o = np.argsort(cols, kind='stable')
    rows, cols = rows[o], cols[o]
    S = np.zeros((n, n), np.float32)
    for c0 in range(0, V, COL_BLOCK):
        a, b = np.searchsorted(cols, [c0, c0 + COL_BLOCK])
        D = np.zeros((n, min(COL_BLOCK, V - c0)), np.float32)
        D[rows[a:b], cols[a:b] - c0] = 1.0
        S += D @ D.T
        del D
    inv = (1.0 / np.sqrt(k)).astype(np.float32)
    S *= inv[:, None]
    S *= inv[None, :]
    iu = np.triu_indices(n, 1)
    up = S[iu]
    cutoff = float(np.quantile(up, 1.0 - TOP_PAIRS))
    sim_q = np.quantile(up, [0.5, 0.9, 0.98, 0.99, 0.999]).tolist()
    del up, iu
    np.fill_diagonal(S, -np.inf)
    fcik = cik[sel]
    is_mapped = np.array([int(c) in mapped_ciks for c in fcik])
    avail = np.array([avail_day(fy, tuple(int(v) for v in fye[i]) if fye[i][0] > 0 else None) for i in sel], np.int64)
    out_i, out_j, out_s, out_rall, out_rmap = [], [], [], [], []
    n_above = np.zeros(n, np.int64)
    for i in range(n):
        js = np.nonzero(S[i] >= cutoff)[0]
        n_above[i] = len(js)
        if not is_mapped[i] or len(js) == 0:
            continue
        js = js[np.argsort(-S[i, js], kind='stable')]
        rmap = np.cumsum(is_mapped[js]) - 1                     # rank among mapped filings (valid where mapped)
        rmap = np.where(is_mapped[js], rmap, -1)
        out_i.append(np.full(len(js), i)); out_j.append(js); out_s.append(S[i, js])
        out_rall.append(np.arange(len(js))); out_rmap.append(rmap)
    del S
    cat = lambda L, dt: np.concatenate(L).astype(dt) if L else np.zeros(0, dt)  # noqa: E731
    pi, pj = cat(out_i, np.int32), cat(out_j, np.int32)
    res = dict(cik=fcik, avail=avail, ntok=ntok[sel], kwords=k.astype(np.int32), mapped=is_mapped, n_above=n_above,
               pi=pi, pj=pj, sim=cat(out_s, np.float32), rank_all=cat(out_rall, np.int32), rank_mapped=cat(out_rmap, np.int32),
               cutoff=np.float64(cutoff), vocab=np.int64(V), n_corpus=np.int64(n_all), n_ciks=np.int64(len(best)))
    np.savez(PEERS / f'{fy}.npz', **res)
    # coverage of the two cap variants (an input check, before any return is computed)
    mi = np.nonzero(is_mapped)[0]
    cap_all = np.bincount(pi[res['rank_all'] < CAP], minlength=n)[mi]
    mapped_j = res['rank_mapped'] >= 0
    cap_all_mapped = np.bincount(pi[(res['rank_all'] < CAP) & mapped_j], minlength=n)[mi]
    cap_map = np.bincount(pi[mapped_j & (res['rank_mapped'] < CAP)], minlength=n)[mi]
    line = (f'FY{fy} | {n_all} | {len(best)} | {n} | {int(is_mapped.sum())} | {V} | {cutoff:.4f} | '
            f'{np.mean(n_above):.0f} / {np.median(n_above):.0f} | {np.mean(n_above[mi] == 0):.0%} | '
            f'{cap_all_mapped.mean():.1f} / {np.median(cap_all_mapped):.0f} | {cap_map.mean():.1f} / {np.median(cap_map):.0f} | '
            + ', '.join(f'{q:.3f}' for q in sim_q))
    return line, cap_all.mean()


def main():
    mapped = set(cik_rows())
    lines = ['FY | corpus filings | CIKs | one per CIK with >= 1000 Item 1 words (n) | of mapped CIKs | vocabulary kept | '
             'cutoff (98th pct) | pairs above cutoff per filing, mean / median | mapped filings with none | '
             'mapped peers per mapped filing, cap among all filings, mean / median | cap among mapped filings, mean / median | '
             'similarity quantiles 50/90/98/99/99.9%']
    for fy in range(FY_FIRST, FY_LAST + 1):
        line, _ = one_year(fy, mapped)
        lines.append(line)
        log(line)
    (HERE / 'peers_counts.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
