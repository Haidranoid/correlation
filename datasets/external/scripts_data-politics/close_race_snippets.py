"""Print sentences about calls/concessions/recounts from close-race Wikipedia pages (for manual review)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW

D = os.path.join(RAW, 'wikipedia_elections')
pages = sys.argv[1:]
MONTH = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
for p in pages:
    fn = os.path.join(D, p.replace(' ', '_') + '.json')
    if not os.path.exists(fn):
        print('==', p, 'MISSING')
        continue
    t = json.load(open(fn))['parse']['wikitext']
    t = re.sub(r'<ref[^>]*/>', '', t)
    t = re.sub(r'<ref.*?</ref>', '', t, flags=re.S)
    t = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', t)
    t = re.sub(r'\{\{[^{}]*\}\}', '', t)
    sents = re.split(r'(?<=[.!?])\s+', t)
    print('==', p)
    n = 0
    for s in sents:
        if re.search(r'conced|called the race|declared .{0,30}winner|projected|recount|certif|AP called|Associated Press', s, re.I) \
                and re.search(MONTH + r'\s+\d{1,2}', s):
            print('   -', ' '.join(s.split())[:330])
            n += 1
            if n >= 9:
                break
