"""The rule engine: per-decision panels, the isotonic calibration, the three rule families under the plan's
accounting (10 slots, no re-equalizing, a sold slot's money buys its replacement) and the turnover-matched random
accounts. Decisions are made with 20-bps thresholds; the cost levels only change what is charged.

Accounting (decided before any result; see the .md's Decisions):
  - $10,000 at the period's first decision, as 10 park slots of $1,000 (cash).
  - A slot holds one stock or is parked: cash (families 1-2, only when no stock can be bought) or SPY (family 3).
    Parked slots are pooled at each decision (all parked money is one SPY sleeve / one cash pile, split equally among
    the parked slots), since they hold the same thing.
  - Between decisions each slot is held as is: value *= the product of its daily returns (holdout.rs `ret`: a missing
    close is a 0 return; with div_fix, pubdata's dividend correction from 2022-07 on). Parked SPY earns Yahoo's
    adjusted-close return.
  - Trades are charged per side on the slot's value: stock c bps, SPY 1 bp (0 at the 0-bps level), cash 0.
  - Nothing is sold at the period's end.
"""
import numpy as np

import common as C

NB = 1000  # percentile bins for the isotonic fit


# ----------------------------------------------------------------------------------------------------------------
# calendars

def spec_of(specs, d):
    for s in specs:
        if s['h0'] - 1 <= d <= s['t'] - 2:
            return s
    raise ValueError(d)


def weekly_calendar(specs, years):
    """Decision days (the graded weeks of these years) and the period's last close: the next year's first graded
    decision, or the last graded week's end for the final year."""
    dec = np.concatenate([s['decisions'] for s in specs if s['year'] in years])
    last = max(years)
    nxt = [s for s in specs if s['year'] == last + 1]
    end = int(nxt[0]['decisions'][0]) if nxt else int(dec[-1] + C.H)
    return dec.astype(np.int64), end


def daily_calendar(specs, years):
    """Every grader decision day h0-1 .. t-2 of these years; the period ends at the last year's t-1."""
    dec = np.concatenate([np.arange(s['h0'] - 1, s['t'] - 1) for s in specs if s['year'] in years])
    last = [s for s in specs if s['year'] == max(years)][0]
    return dec.astype(np.int64), int(last['t'] - 1)


# ----------------------------------------------------------------------------------------------------------------
# percentiles and calibration

def avg_rank_pct(x):
    """Percentile in (0, 1) of each value by average rank (ties share their mean rank): (rank - 0.5) / n."""
    n = len(x)
    o = np.argsort(x, kind='mergesort')
    xs = x[o]
    r = np.empty(n)
    # average ranks over tied groups
    starts = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1]
    ends = np.r_[starts[1:], n]
    ranks = np.empty(n)
    for a, b in zip(starts, ends):
        ranks[a:b] = 0.5 * (a + b - 1) + 1.0
    r[o] = ranks
    return (r - 0.5) / n


def pava(y, w):
    """Weighted isotonic (non-decreasing) regression by pool-adjacent-violators."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        if wi <= 0:
            vals.append(np.nan); wts.append(0.0); cnt.append(1)
            continue
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1:
            # find previous block with weight
            j = len(vals) - 2
            while j >= 0 and wts[j] == 0:
                j -= 1
            if j < 0 or vals[j] <= vals[-1]:
                break
            # merge block j..end into one
            tw = sum(wts[j:]); tv = sum(v * ww for v, ww in zip(vals[j:], wts[j:]) if ww > 0) / tw
            tc = sum(cnt[j:])
            del vals[j:], wts[j:], cnt[j:]
            vals.append(tv); wts.append(tw); cnt.append(tc)
    out = np.concatenate([np.full(c, v) for v, c in zip(vals, cnt)])
    # bins with no weight take the next fitted value to the left (or right at the start)
    ok = np.isfinite(out)
    idx = np.where(ok, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    first = np.flatnonzero(ok)[0]
    out[:first] = out[first]
    return out


def window_products(R, d_lo, n_days):
    """Cumulative log-growth table: L[i] = sum_{u < d_lo + i} log(1 + r(u)) for the Returns object R (rows from
    R.d_lo). Window product over decision days [a, b) = exp(L[b - d_lo] - L[a - d_lo])."""
    lr = np.log(np.maximum(1.0 + R.r[d_lo - R.d_lo:d_lo - R.d_lo + n_days], 1e-12))
    L = np.zeros((n_days + 1, lr.shape[1]))
    np.cumsum(lr, axis=0, out=L[1:])
    return L


def scored_set(specs, scores, d):
    s = spec_of(specs, d)
    u = s['universe']
    sc = np.asarray(scores[u, d], dtype=np.float64)
    ok = np.isfinite(sc)
    return u[ok], sc[ok]


def calibrate(E, specs, scores, R, days_fit, step, kmax, end):
    """g_k for k = 1..kmax: isotonic map from the day's score percentile (scored set: the year's universe with a finite
    score) to the market-relative return over days (d + step*(k-1), d + step*k], market = the scored set's
    equal-weight mean simple return over the same days. Fitted on every day in days_fit whose window ends by `end`."""
    lo = int(days_fit[0])
    L = window_products(R, lo, end - lo)
    sums = np.zeros((kmax, NB)); cnts = np.zeros((kmax, NB))
    for d in days_fit:
        S, sc = scored_set(specs, scores, int(d))
        if len(S) < 20:
            continue
        b = np.minimum((avg_rank_pct(sc) * NB).astype(int), NB - 1)
        for k in range(1, kmax + 1):
            a, e = int(d) + step * (k - 1), int(d) + step * k
            if e > end:
                continue
            g = np.exp(L[e - lo, S] - L[a - lo, S]) - 1.0
            rel = g - g.mean()
            sums[k - 1] += np.bincount(b, weights=rel, minlength=NB)
            cnts[k - 1] += np.bincount(b, minlength=NB)
    g = np.zeros((kmax, NB)); raw = np.zeros((kmax, NB))
    for k in range(kmax):
        raw[k] = np.where(cnts[k] > 0, sums[k] / np.maximum(cnts[k], 1), np.nan)
        g[k] = pava(np.nan_to_num(raw[k]), cnts[k])
    return {'g': g, 'raw': raw, 'counts': cnts, 'step': step}


def rank_ic(E, specs, scores, R, days_eval, step, kmax, end):
    """Mean daily Spearman IC of the score against the market-relative return in window k (k = 1..kmax)."""
    lo = int(days_eval[0])
    L = window_products(R, lo, end - lo)
    ics = [[] for _ in range(kmax)]
    for d in days_eval:
        S, sc = scored_set(specs, scores, int(d))
        if len(S) < 20:
            continue
        rs = avg_rank_pct(sc)
        for k in range(1, kmax + 1):
            a, e = int(d) + step * (k - 1), int(d) + step * k
            if e > end:
                continue
            g = np.exp(L[e - lo, S] - L[a - lo, S]) - 1.0
            rg = avg_rank_pct(g)
            ics[k - 1].append(np.corrcoef(rs, rg)[0, 1])
    out = []
    for k in range(kmax):
        x = np.array(ics[k])
        out.append({'k': k + 1, 'n': len(x), 'ic': float(x.mean()), 't': float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
                    'days_pos': int((x > 0).sum())})
    return out


# ----------------------------------------------------------------------------------------------------------------
# panels

class Panel:
    """Per-decision data for one scores file on one calendar: the scored set sorted by score (descending, ties by
    column), each member's percentile in the full scored set and G_h, a stock -> position lookup."""

    def __init__(self, E, specs, scores, decisions, end, calib=None, hs=(1, 2, 4), large500=False):
        self.N = E['close'].shape[0]
        self.decisions = np.asarray(decisions, dtype=np.int64)
        self.end = int(end)
        self.hs = hs
        self.S, self.sc, self.G, self.where = [], [], [], []
        dv = E['dollar_volume']
        for d in self.decisions:
            d = int(d)
            S, sc = scored_set(specs, scores, d)
            pct = avg_rank_pct(sc) if len(S) else np.zeros(0)
            if large500 and len(S) > 500:
                med = np.nanmedian(np.asarray(dv[S, d - 62:d + 1], dtype=np.float64), axis=1)
                med = np.where(np.isfinite(med), med, -1.0)
                keep = np.sort(np.argsort(-med, kind='mergesort')[:500])
                S, sc, pct = S[keep], sc[keep], pct[keep]
            o = np.lexsort((S, -sc))
            S, sc, pct = S[o], sc[o], pct[o]
            self.S.append(S.astype(np.int64))
            self.sc.append(sc)
            if calib is not None:
                b = np.minimum((pct * NB).astype(int), NB - 1)
                cg = np.cumsum(calib['g'], axis=0)  # G_h = sum_{k<=h} g_k
                self.G.append({h: cg[h - 1][b] for h in hs})
            w = np.full(self.N, -1, dtype=np.int32)
            w[S] = np.arange(len(S), dtype=np.int32)
            self.where.append(w)

    def __len__(self):
        return len(self.decisions)


def period_products(R, panel, spy_r):
    """For each decision k: the product of (1 + r) over days [d_k, d_{k+1}) for every stock [N], and SPY's."""
    ends = np.r_[panel.decisions[1:], panel.end]
    L = np.zeros((1, R.r.shape[1]))
    P, Ps = [], []
    for a, e in zip(panel.decisions, ends):
        a, e = int(a), int(e)
        P.append(np.prod(1.0 + R.r[a - R.d_lo:e - R.d_lo], axis=0))
        Ps.append(float(np.prod(1.0 + spy_r[a:e])))
    return P, np.array(Ps)


# ----------------------------------------------------------------------------------------------------------------
# rules: each returns, for the current slots (stock index or -1 = parked), the new slots and per-slot transitions

KEEP, SS, SP, PS, PP = 0, 1, 2, 3, 4


def _fill(new, trans, free, cands, park_to):
    """Put candidates (in order) into the free slots (in order); slots left over are parked."""
    ci = 0
    for i in free:
        was_stock = new[i] >= 0
        if ci < len(cands):
            new[i] = cands[ci]; ci += 1
            trans[i] = SS if was_stock else PS
        else:
            new[i] = -1
            trans[i] = SP if was_stock else PP
    return ci


def _cands(S, sc, held, need, G=None, thr=None):
    """The first `need` unheld stocks in score order (S is sorted by score, descending; G is non-increasing along it
    because every g_k is non-decreasing in the percentile) with score > 0, or with G > thr when thr is given."""
    out = []
    for p in range(len(S)):
        if len(out) >= need:
            break
        if thr is not None:
            if not G[p] > thr:
                break
        elif not sc[p] > 0:
            continue
        j = int(S[p])
        if j not in held:
            out.append((p, j))
    return out


def rule_buffer(K):
    def f(slots, P, k):
        S, sc, where = P.S[k], P.sc[k], P.where[k]
        new = slots.copy(); trans = np.full(10, KEEP)
        free = []
        for i in range(10):
            j = slots[i]
            if j < 0:
                free.append(i); trans[i] = PP
            elif where[j] < 0 or where[j] >= K:
                free.append(i)
        held = set(int(j) for j in slots if j >= 0)
        cands = [j for _, j in _cands(S, sc, held, len(free))]
        _fill(new, trans, free, cands, None)
        return new, trans
    return f


def rule_switch(h, m):
    thr = (1.0 + m) * C.ROUND_TRIP

    def f(slots, P, k):
        S, sc, where, G = P.S[k], P.sc[k], P.where[k], P.G[k][h]
        new = slots.copy(); trans = np.full(10, KEEP)
        free = [i for i in range(10) if slots[i] < 0 or where[slots[i]] < 0]
        for i in range(10):
            if slots[i] < 0:
                trans[i] = PP
        held = set(int(j) for j in slots if j >= 0)
        cl = _cands(S, sc, held, len(free) + 10)
        used = _fill(new, trans, free, [j for _, j in cl], None)
        cl = cl[used:]
        # switches: holdings by G ascending (ties: the lower score first), paired in turn with the best candidates;
        # stop at the first pair whose gain does not beat (1 + m) x the round trip
        hold = [i for i in range(10) if i not in free]
        hold.sort(key=lambda i: (G[where[slots[i]]], -where[slots[i]]))
        for i, (p, j) in zip(hold, cl):
            if G[p] - G[where[slots[i]]] > thr:
                new[i] = j; trans[i] = SS
            else:
                break
        return new, trans
    return f


def rule_confident(h, c):
    thr = c * C.ROUND_TRIP

    def f(slots, P, k):
        S, sc, where, G = P.S[k], P.sc[k], P.where[k], P.G[k][h]
        new = slots.copy(); trans = np.full(10, KEEP)
        sold = [i for i in range(10) if slots[i] >= 0 and (where[slots[i]] < 0 or G[where[slots[i]]] < 0)]
        parked = [i for i in range(10) if slots[i] < 0]
        held = set(int(j) for j in slots if j >= 0)
        cands = [j for _, j in _cands(S, sc, held, len(sold) + len(parked), G=G, thr=thr)]
        used = _fill(new, trans, sold, cands, None)
        _fill(new, trans, parked, cands[used:], None)
        return new, trans
    return f


def settings(hs=(1, 2, 4)):
    """The 19 graded settings (K = 10, the baseline, excluded). hs: the three horizons (weeks, or days for the daily
    signal: (1, 2, 5))."""
    return ([('buffer', {'K': K}) for K in (20, 30, 50, 100)]
            + [('switch', {'h': h, 'm': m}) for h in hs for m in (0.0, 0.5, 1.0)]
            + [('confident', {'h': h, 'c': c}) for h in hs for c in (1, 2)])


SETTINGS = settings()
BASELINE = ('buffer', {'K': 10})


def make_rule(fam, p):
    return {'buffer': lambda: rule_buffer(p['K']), 'switch': lambda: rule_switch(p['h'], p['m']),
            'confident': lambda: rule_confident(p['h'], p['c'])}[fam]()


def label(fam, p):
    if fam == 'buffer':
        return f"buffer K={p['K']}"
    if fam == 'switch':
        return f"switch h={p['h']} m={p['m']:g}"
    return f"confident h={p['h']} c={p['c']:g}"


# ----------------------------------------------------------------------------------------------------------------
# accounts

def _mult(trans, costs, park_cost):
    """Cost multiplier per transition and cost level: [ncost, 5] (KEEP, SS, SP, PS, PP)."""
    out = []
    for c in costs:
        cs = c / 1e4
        p = (park_cost if c > 0 else 0.0) / 1e4
        out.append([1.0, 1 - 2 * cs, 1 - cs - p, 1 - p - cs, 1.0])
    return np.array(out)


def simulate(panel, rule, P, Pspy, costs=(0.0, 5.0, 20.0), park='cash', avg=None):
    """One strategy account. Returns values at each decision (before trades) and at the end, per cost; per-week
    counts of SS/SP/PS transitions; names held; dollar turnover (traded value / account value, both sides)."""
    nc = len(costs)
    pc = C.SPY_COST if park == 'spy' else 0.0
    M = _mult(None, costs, pc)
    M0 = M.copy()
    M0[:, PS] = 1 - np.array(costs) / 1e4                      # cash -> stock at the start
    M0[:, PP] = 1 - np.array([(pc if c > 0 else 0) for c in costs]) / 1e4  # cash -> SPY at the start
    slots = np.full(10, -1, dtype=np.int64)
    val = np.full((nc, 10), 1000.0)
    n = len(panel)
    V = np.zeros((nc, n + 1)); ev = np.zeros((n, 3), dtype=np.int64); nheld = np.zeros(n, dtype=np.int64)
    dturn = np.zeros((nc, n)); hist = []
    for k in range(n):
        V[:, k] = val.sum(axis=1)
        pk = slots < 0
        if pk.any():
            val[:, pk] = val[:, pk].mean(axis=1, keepdims=True)
        new, trans = rule(slots, panel, k)
        mm = M0 if k == 0 else M
        tv = np.zeros(nc)
        for i in range(10):
            t = trans[i]
            if t == KEEP:
                continue
            if t in (SS,):
                tv += 2 * val[:, i]
            elif t in (SP, PS):
                tv += val[:, i]
            elif t == PP and k == 0 and park == 'spy':
                pass
            val[:, i] *= mm[:, t]
        dturn[:, k] = tv / np.maximum(V[:, k], 1e-12)
        ev[k] = [(trans == SS).sum(), (trans == SP).sum(), (trans == PS).sum()]
        slots = new
        nheld[k] = (slots >= 0).sum()
        hist.append(slots.copy())
        g = np.where(slots >= 0, P[k][np.maximum(slots, 0)], Pspy[k] if park == 'spy' else 1.0)
        val *= g[None, :]
    V[:, n] = val.sum(axis=1)
    return {'V': V, 'events': ev, 'nheld': nheld, 'dturn': dturn, 'slots': np.array(hist)}


def simulate_random(panel, events, P, Pspy, R=1000, seed=20261021, costs=(0.0, 5.0, 20.0), park='cash'):
    """R turnover-matched random accounts (the plan's null): each week, as many stock->stock replacements, stock->park
    and park->stock moves as the strategy made; sales uniformly random among the account's stocks (stocks that lost
    their score first), purchases uniformly random from that day's scored set (not held, not just sold). Parked money
    is pooled as for the strategy. Same accounting and costs. Returns values [ncost, R, n+1]."""
    rng = np.random.default_rng(seed)
    nc = len(costs)
    pc = C.SPY_COST if park == 'spy' else 0.0
    M = _mult(None, costs, pc)
    cs = np.array(costs) / 1e4
    pcs = np.array([(pc if c > 0 else 0) for c in costs]) / 1e4
    N = panel.N
    slots = np.full((R, 10), -1, dtype=np.int64)
    val = np.full((nc, R, 10), 1000.0)
    n = len(panel)
    V = np.zeros((nc, R, n + 1))
    rows = np.arange(R)[:, None]
    for k in range(n):
        V[:, :, k] = val.sum(axis=2)
        pk = slots < 0
        if pk.any():
            tot = np.where(pk[None], val, 0).sum(axis=2, keepdims=True)
            cnt = np.maximum(pk.sum(axis=1), 1)[None, :, None]
            val = np.where(pk[None], tot / cnt, val)
        n_ss, n_sp, n_ps = (int(x) for x in events[k])
        S = panel.S[k]
        where = panel.where[k]
        first = k == 0
        sold_stocks = np.full((R, 0), -1, dtype=np.int64)
        if n_ss + n_sp > 0:
            unscored = (slots >= 0) & (where[np.maximum(slots, 0)] < 0)
            key = rng.random((R, 10)) - 10.0 * unscored + 100.0 * (slots < 0)
            order = np.argsort(key, axis=1)
            sell = order[:, :n_ss + n_sp]
            sold_stocks = slots[rows, sell]
            ss_slots, sp_slots = sell[:, :n_ss], sell[:, n_ss:]
            if n_sp:
                for c in range(nc):
                    val[c][rows, sp_slots] *= 1 - cs[c] - pcs[c]
                slots[rows, sp_slots] = -1
        else:
            ss_slots = np.zeros((R, 0), dtype=np.int64)
        ps_slots = np.zeros((R, 0), dtype=np.int64)
        if n_ps > 0:
            key = rng.random((R, 10)) + 100.0 * (slots >= 0)
            # don't take a slot that was just parked this week (the strategy never does both in one week)
            ps_slots = np.argsort(key, axis=1)[:, :n_ps]
        buy_slots = np.concatenate([ss_slots, ps_slots], axis=1)
        nb = buy_slots.shape[1]
        if nb:
            draws = S[rng.integers(0, len(S), size=(R, nb))]
            for _ in range(1000):
                held_now = slots.copy()
                held_now[rows, buy_slots] = -2
                bad = (draws[:, :, None] == held_now[:, None, :]).any(axis=2)
                bad |= (draws[:, :, None] == sold_stocks[:, None, :]).any(axis=2)
                for a in range(nb):  # duplicates within the row
                    for b in range(a):
                        bad[:, a] |= draws[:, a] == draws[:, b]
                if not bad.any():
                    break
                draws[bad] = S[rng.integers(0, len(S), size=int(bad.sum()))]
            for c in range(nc):
                if n_ss:
                    val[c][rows, ss_slots] *= 1 - 2 * cs[c]
                if n_ps:
                    if first:
                        val[c][rows, ps_slots] *= 1 - cs[c]
                    else:
                        val[c][rows, ps_slots] *= 1 - pcs[c] - cs[c]
            slots[rows, buy_slots] = draws
        if first and park == 'spy':
            pk = slots < 0
            for c in range(nc):
                val[c] = np.where(pk, val[c] * (1 - pcs[c]), val[c])
        Pext = np.r_[P[k], Pspy[k] if park == 'spy' else 1.0]
        g = Pext[np.where(slots >= 0, slots, N)]
        val *= g[None]
    V[:, :, n] = val.sum(axis=2)
    return V


# ----------------------------------------------------------------------------------------------------------------
# summaries

def cagr(v0, v1, ndays):
    return (v1 / v0) ** (252.0 / ndays) - 1.0


def avg_stock_products(panel, P):
    return np.array([P[k][panel.S[k]].mean() for k in range(len(panel))])


def tstat(x):
    x = np.asarray(x)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def year_marks(panel, specs, years):
    """Index (into the decision list) of each year's first decision, plus the end (len(panel))."""
    marks = []
    for y in years:
        s = [s for s in specs if s['year'] == y][0]
        k = int(np.searchsorted(panel.decisions, s['h0'] - 1))
        marks.append(k)
    return marks + [len(panel)]


def day_at(panel, k):
    return int(panel.decisions[k]) if k < len(panel) else panel.end


def summarize(res, panel, avgP, spy_adj, specs, years, costs=(0.0, 5.0, 20.0)):
    V = res['V']
    n = len(panel)
    ndays = panel.end - int(panel.decisions[0])
    d0, d1 = int(panel.decisions[0]), panel.end
    spy_total = spy_adj[d1] / spy_adj[d0] - 1.0
    out = {'final': {f'{c:g}': float(V[i, -1]) for i, c in enumerate(costs)},
           'cagr': {f'{c:g}': float(cagr(V[i, 0], V[i, -1], ndays)) for i, c in enumerate(costs)},
           'spy_final': float(1e4 * (1 + spy_total)), 'spy_cagr': float(cagr(1.0, 1 + spy_total, ndays)),
           'weeks': n, 'days': ndays}
    ev = res['events']
    out['names_per_week'] = float(ev[1:].sum(axis=1).mean())
    out['stock_names_per_week'] = float((2 * ev[1:, 0] + ev[1:, 1] + ev[1:, 2]).mean() / 2)
    yrs = ndays / 252.0
    ci = costs.index(20.0)
    out['dollar_turnover_per_year'] = float(res['dturn'][ci, 1:].sum() / yrs)  # both sides, x account value
    out['mean_stocks_held'] = float(res['nheld'].mean())
    wk = V[ci, 1:] / V[ci, :-1] - 1.0
    ex = wk - (avgP - 1.0)
    out['weekly_excess_20'] = float(ex.mean()); out['weekly_t_20'] = tstat(ex)
    wk0 = V[0, 1:] / V[0, :-1] - 1.0
    ex0 = wk0 - (avgP - 1.0)
    out['weekly_excess_0'] = float(ex0.mean()); out['weekly_t_0'] = tstat(ex0)
    marks = year_marks(panel, specs, years)
    by = []
    for y, a, b in zip(years, marks[:-1], marks[1:]):
        da, db = day_at(panel, a), day_at(panel, b)
        row = {'year': y, 'spy': float(spy_adj[db] / spy_adj[da] - 1.0),
               'avg_stock': float(np.prod(avgP[a:b]) - 1.0)}
        for i, c in enumerate(costs):
            row[f'ret_{c:g}'] = float(V[i, b] / V[i, a] - 1.0)
        row['names_per_week'] = float(ev[max(a, 1):b].sum(axis=1).mean()) if b > max(a, 1) else float('nan')
        by.append(row)
    out['by_year'] = by
    out['after_costs_5_minus_spy'] = float(np.mean([r['ret_5'] - r['spy'] for r in by]))
    out['mean_yearly'] = {f'{c:g}': float(np.mean([r[f'ret_{c:g}'] for r in by])) for c in costs}
    out['spy_mean_yearly'] = float(np.mean([r['spy'] for r in by]))
    return out


def simulate_shadow(panel, slots_hist, P, Pspy, R=1000, seed=20261021, costs=(0.0, 5.0, 20.0), park='cash'):
    """Robustness null: each random account shadows the strategy slot by slot (when the strategy changes slot i's
    stock, the account replaces its slot i with a uniformly random scored stock; parks and un-parks with it). Same
    turnover, holding periods and stock/park split as the strategy, slot for slot."""
    rng = np.random.default_rng(seed)
    nc = len(costs)
    pc = C.SPY_COST if park == 'spy' else 0.0
    cs = np.array(costs) / 1e4
    pcs = np.array([(pc if c > 0 else 0) for c in costs]) / 1e4
    N = panel.N
    slots = np.full((R, 10), -1, dtype=np.int64)
    val = np.full((nc, R, 10), 1000.0)
    n = len(panel)
    V = np.zeros((nc, R, n + 1))
    prev = np.full(10, -1, dtype=np.int64)
    for k in range(n):
        V[:, :, k] = val.sum(axis=2)
        pk = slots < 0
        if pk.any():
            tot = np.where(pk[None], val, 0).sum(axis=2, keepdims=True)
            cnt = np.maximum(pk.sum(axis=1), 1)[None, :, None]
            val = np.where(pk[None], tot / cnt, val)
        cur = slots_hist[k]
        S = panel.S[k]
        for i in range(10):
            a, b = prev[i], cur[i]
            if a == b:
                continue
            if b >= 0:
                draw = S[rng.integers(0, len(S), size=R)]
                for _ in range(1000):
                    others = np.delete(slots, i, axis=1)
                    bad = (draw[:, None] == others).any(axis=1) | (draw == slots[:, i])
                    if not bad.any():
                        break
                    draw[bad] = S[rng.integers(0, len(S), size=int(bad.sum()))]
                for c in range(nc):
                    if a >= 0:
                        val[c][:, i] *= 1 - 2 * cs[c]
                    elif k == 0:
                        val[c][:, i] *= 1 - cs[c]
                    else:
                        val[c][:, i] *= 1 - pcs[c] - cs[c]
                slots[:, i] = draw
            else:
                for c in range(nc):
                    val[c][:, i] *= 1 - cs[c] - pcs[c]
                slots[:, i] = -1
        if k == 0 and park == 'spy':
            for i in range(10):
                if cur[i] < 0:
                    for c in range(nc):
                        val[c][:, i] *= 1 - pcs[c]
        prev = cur.copy()
        Pext = np.r_[P[k], Pspy[k] if park == 'spy' else 1.0]
        g = Pext[np.where(slots >= 0, slots, N)]
        val *= g[None]
    V[:, :, n] = val.sum(axis=2)
    return V
