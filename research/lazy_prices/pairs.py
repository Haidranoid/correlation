"""Year-over-year similarity of each mapped company's 10-K: cosine similarity of this year's and last year's
term-frequency vectors, over Items 1, 1A and 7 concatenated, and per section.

- One filing per company and corpus year: the one with the most words over the three sections.
- A section counts when it has at least 100 words in both filings; the concatenation uses those sections only.
- Pairs are consecutive corpus (fiscal) years of the same CIK. When both filings' fiscal-year ends are known, pairs
  more than 15 or fewer than 9 months apart are dropped.
- The similarity is known from the later filing's date under the imposters experiment's rule (common.avail_day).

    python3 -I pairs.py        # reads vec/*.npz, writes pairs.npz in the work folder and pairs_counts.txt here
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import (HERE, MAX_GAP_MONTHS, MIN_GAP_MONTHS, MIN_SECTION_WORDS, SECTIONS, VEC, WORK, avail_day,  # noqa: E402
                    np)


class Year:
    def __init__(self, y):
        z = np.load(VEC / f'{y}.npz')
        self.y = y
        self.n_corpus = int(z['n_corpus'])
        self.cik, self.fye = z['cik'], z['fye']
        self.n_raw = len(self.cik)
        self.sec = {s: (z[f'{s}_ids'], z[f'{s}_cnt'], z[f'{s}_ptr'], z[f'{s}_words']) for s in SECTIONS}
        tot = sum(self.sec[s][3] for s in SECTIONS)
        best = {}
        for i, c in enumerate(self.cik):
            c = int(c)
            if c not in best or tot[i] > tot[best[c]]:
                best[c] = i
        self.by_cik = best
        self.year_field = z['year']

    def words(self, i, s):
        return int(self.sec[s][3][i])

    def vector(self, i, secs):
        ids, cnts = [], []
        for s in secs:
            a, b = self.sec[s][2][i], self.sec[s][2][i + 1]
            ids.append(self.sec[s][0][a:b]); cnts.append(self.sec[s][1][a:b])
        if len(secs) == 1:
            return ids[0], cnts[0].astype(np.float64)
        ids, cnts = np.concatenate(ids), np.concatenate(cnts).astype(np.float64)
        u, inv = np.unique(ids, return_inverse=True)
        return u, np.bincount(inv, weights=cnts, minlength=len(u))


def cosine(a, b):
    (ia, ca), (ib, cb) = a, b
    _, x, y = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
    na, nb = np.sqrt((ca * ca).sum()), np.sqrt((cb * cb).sum())
    if na == 0 or nb == 0:
        return np.nan
    return float((ca[x] * cb[y]).sum() / (na * nb))


def main():
    years = sorted(int(p.stem) for p in VEC.glob('[0-9][0-9][0-9][0-9].npz'))
    rows = []
    lines = ['year | corpus filings | of mapped CIKs | after one per CIK | pairs with last year | dropped: no shared '
             'section | dropped: fiscal-year ends apart | kept | Item 1 / 1A / 7 shared | non-December fiscal year']
    prev = None
    for y in years:
        cur = Year(y)
        bad_year = int((cur.year_field != y).sum())
        if prev is not None and prev.y == y - 1:
            n_pair = n_noshare = n_gap = 0
            shared_n = {s: 0 for s in SECTIONS}
            non_dec = 0
            for c, i in cur.by_cik.items():
                j = prev.by_cik.get(c)
                if j is None:
                    continue
                n_pair += 1
                f1, f0 = cur.fye[i], prev.fye[j]
                gap = np.nan
                if f1[0] > 0 and f0[0] > 0:
                    gap = (f1[0] * 12 + f1[1]) - (f0[0] * 12 + f0[1])
                    if gap > MAX_GAP_MONTHS or gap < MIN_GAP_MONTHS:
                        n_gap += 1
                        continue
                shared = [s for s in SECTIONS if cur.words(i, s) >= MIN_SECTION_WORDS and prev.words(j, s) >= MIN_SECTION_WORDS]
                if not shared:
                    n_noshare += 1
                    continue
                per = {s: (cosine(cur.vector(i, [s]), prev.vector(j, [s])) if s in shared else np.nan) for s in SECTIONS}
                sim = per[shared[0]] if len(shared) == 1 else cosine(cur.vector(i, shared), prev.vector(j, shared))
                for s in shared:
                    shared_n[s] += 1
                fye = tuple(int(v) for v in f1) if f1[0] > 0 else None
                month = int(f1[1]) if f1[0] > 0 else -1
                non_dec += month not in (12, -1)
                # sensitivity fixed before results: every non-empty section of each filing, concatenated as is
                ne_c = [s for s in SECTIONS if cur.words(i, s) > 0]
                ne_p = [s for s in SECTIONS if prev.words(j, s) > 0]
                sim_all = cosine(cur.vector(i, ne_c), prev.vector(j, ne_p))
                rows.append((c, y, avail_day(y, fye), sim, per['section_1'], per['section_1A'], per['section_7'],
                             month, gap, sum(cur.words(i, s) for s in shared), sum(prev.words(j, s) for s in shared),
                             avail_day(y, fye, 'april'), sim_all))
            kept = n_pair - n_noshare - n_gap
            lines.append(f'{y} | {cur.n_corpus} | {cur.n_raw} | {len(cur.by_cik)} | {n_pair} | {n_noshare} | {n_gap} | '
                         f'{kept} | {shared_n["section_1"]} / {shared_n["section_1A"]} / {shared_n["section_7"]} | '
                         f'{non_dec}' + (f' (year field differs: {bad_year})' if bad_year else ''))
        else:
            lines.append(f'{y} | {cur.n_corpus} | {cur.n_raw} | {len(cur.by_cik)} | (first year)')
        print(lines[-1], flush=True)
        prev = cur
    a = np.array(rows, dtype=np.float64)
    out = {'cik': a[:, 0].astype(np.int64), 'fy': a[:, 1].astype(np.int32), 'avail': a[:, 2].astype(np.int64),
           'sim': a[:, 3], 'sim_1': a[:, 4], 'sim_1A': a[:, 5], 'sim_7': a[:, 6], 'fye_month': a[:, 7].astype(np.int32),
           'gap_months': a[:, 8], 'words_cur': a[:, 9], 'words_prev': a[:, 10],
           'avail_april': a[:, 11].astype(np.int64), 'sim_all': a[:, 12]}
    np.savez(WORK / 'pairs.npz', **out)
    sim = out['sim']
    lines.append('')
    lines.append(f'{len(sim)} pairs kept; similarity quantiles (5, 25, 50, 75, 95%): '
                 + ', '.join(f'{q:.3f}' for q in np.nanpercentile(sim, [5, 25, 50, 75, 95])))
    m = out['fye_month']
    lines.append(f'fiscal-year end month known for {int((m > 0).sum())}; non-December {int(((m > 0) & (m != 12)).sum())} '
                 f'({((m > 0) & (m != 12)).sum() / max((m > 0).sum(), 1):.1%} of known)')
    (HERE / 'pairs_counts.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[-2:]))


if __name__ == '__main__':
    main()
