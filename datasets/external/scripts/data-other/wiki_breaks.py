import json, os, glob, statistics, sys
W = os.path.expanduser("~/.local/share/correlation2/external/raw/wikimedia")
for p in sorted(glob.glob(os.path.join(W, "views_*.json"))):
    items = json.load(open(p))["items"]
    v = [it["views"] for it in items]; d = [it["timestamp"][:8] for it in items]
    # 30-day block medians; report max ratio between consecutive non-overlapping blocks
    blocks = [(d[i], statistics.median(v[i:i+30])) for i in range(0, len(v) - 29, 30)]
    worst = max(((max(a[1],1)/max(b[1],1) if a[1] > b[1] else max(b[1],1)/max(a[1],1)), a[0], b[0], a[1], b[1])
                for a, b in zip(blocks, blocks[1:]))
    print(f"{os.path.basename(p)[6:-5]:34s} n={len(v)} first={d[0]} last={d[-1]} min={min(v)} "
          f"blk_first={blocks[0][1]:.0f} blk_last={blocks[-1][1]:.0f} max_block_jump={worst[0]:.1f}x at {worst[2]} ({worst[3]:.0f}->{worst[4]:.0f})")
