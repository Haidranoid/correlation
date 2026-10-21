"""Same-day correlation of external series with US-listed stocks (store, read-only): positive-control candidates."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dmlib
from probe import load, to_days, moves, corr

STORE = os.path.expanduser('~/.local/share/correlation2/stores/stooq-2025-06-27')
syms = {}
for line in open(os.path.join(STORE, 'symbols.tsv')).read().splitlines()[1:]:
    p = line.split('\t')
    if p[3] == 'stock':
        syms.setdefault(p[1], int(p[0]))
buf = np.memmap(os.path.join(STORE, 'prices.bin'), dtype=np.uint8, mode='r')
n = int(np.frombuffer(buf[12:16].tobytes(), '<u4')[0]); rows = int(np.frombuffer(buf[16:24].tobytes(), '<u8')[0])
off = np.frombuffer(buf[24:24 + 8 * (n + 1)].tobytes(), '<u8').astype(np.int64)
pos = 24 + 8 * (n + 1); cols = []
for _ in range(6):
    pos = -(-pos // 64) * 64; cols.append(pos); pos += rows * 4
def stock(t):
    i = syms[t]; a, b = off[i], off[i + 1]
    d = np.frombuffer(buf[cols[0] + 4 * a:cols[0] + 4 * b].tobytes(), '<i4')
    c = np.frombuffer(buf[cols[4] + 4 * a:cols[4] + 4 * b].tobytes(), '<f4').astype(float)
    return [(np.datetime64('1970-01-01') + np.timedelta64(int(x), 'D')).astype(str) for x in d], c
cat = {r['id']: r for r in csv.DictReader(open(dmlib.CATALOG))}
PAIRS = [
    ('fred_dcoilwtico', ['XOM', 'CVX', 'COP', 'OXY', 'SLB', 'APA'], '2005'),
    ('etf_uso', ['XOM', 'OXY', 'APA'], '2007'),
    ('fred_dhhngsp', ['EQT', 'RRC', 'AR'], '2005'),
    ('etf_ung', ['EQT', 'RRC', 'AR'], '2008'),
    ('stooq_xauusd', ['NEM', 'AEM', 'GOLD', 'KGC', 'RGLD'], '2005'),
    ('etf_gld', ['NEM', 'AEM', 'GOLD', 'KGC'], '2005'),
    ('stooq_xagusd', ['PAAS', 'HL', 'AG'], '2005'),
    ('etf_cper', ['FCX', 'SCCO'], '2012'),
    ('etf_ura', ['CCJ', 'UEC'], '2011'),
    ('fred_dgs10', ['JPM', 'BAC', 'SCHW', 'NEE', 'O'], '2005'),
    ('fred_t10y2y', ['JPM', 'BAC', 'KEY'], '2005'),
    ('fred_vixcls', ['AAPL', 'JPM', 'CAT'], '2005'),
    ('stooq_btc', ['COIN', 'MSTR', 'MARA', 'RIOT'], '2021'),
    ('fred_cbbtcusd', ['COIN', 'MSTR', 'MARA', 'RIOT'], '2021'),
    ('stooq_eth', ['COIN'], '2021'),
    ('stooq_taiex', ['TSM'], '2005'),
    ('stooq_nikkei225', ['TM', 'SONY', 'MUFG'], '2005'),
    ('stooq_hangseng', ['BABA', 'JD', 'BIDU'], '2015'),
    ('stooq_dax', ['SAP', 'DB'], '2005'),
    ('stooq_ftse100', ['BP', 'SHEL', 'AZN', 'HSBC'], '2005'),
    ('stooq_bovespa', ['PBR', 'VALE', 'ITUB'], '2005'),
    ('stooq_sensex', ['INFY', 'HDB', 'IBN'], '2005'),
    ('fred_dexmxus', ['AMX'], '2005'),
    ('etf_dba', ['ADM', 'BG'], '2007'),
    ('etf_corn', ['ADM', 'BG', 'DE'], '2011'),
]
for sid, tickers, start in PAIRS:
    d, v = load(sid); tr = cat[sid]['transform']
    out = []
    for t in tickers:
        if t not in syms:
            out.append(f"{t}:n/a"); continue
        sd, sc = stock(t)
        days = [x for x in sd if f'{start}-01-01' <= x <= '2025-06-27']
        if len(days) < 300:
            out.append(f"{t}:short"); continue
        st = moves(to_days(sd, sc, days), 'logdiff'); x = moves(to_days(d, v, days), tr)
        c0, nn = corr(x, st); cm1, _ = corr(x[1:], st[:-1]); cp1, _ = corr(x[:-1], st[1:])
        out.append(f"{t}:{c0:+.2f}(next{cp1:+.2f},prev{cm1:+.2f})")
    print(f"{sid:<20} kad={cat[sid]['known_after_days']} from {start}: " + "  ".join(out))
