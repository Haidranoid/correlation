"""Parse the summary tables of the yearly 'YYYY United States gubernatorial elections' Wikipedia pages.

Output work/governor_races.json: list of races
  {year, state, incumbent_party, winner_party, winner_pct, runnerup_party, runnerup_pct, margin_rd, result_text, special}
margin_rd = (R share - D share) in percentage points when both a Republican and a Democrat are among the
candidates listed, else None.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, WORK

D = os.path.join(RAW, 'wikipedia_elections')
STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida',
          'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
          'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
          'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
          'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
          'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']


def clean(s):
    s = re.sub(r'<ref[^>]*/>', '', s)
    s = re.sub(r'<ref.*?</ref>', '', s, flags=re.S)
    s = re.sub(r'\{\{efn.*?\}\}', '', s, flags=re.S)
    return s


def party_word(s):
    s2 = s.lower()
    if 'republican' in s2:
        return 'R'
    if 'democrat' in s2 or 'dfl' in s2 or 'democratic–farmer' in s2:
        return 'D'
    if 'independen' in s2 or 'reform' in s2 or 'libertarian' in s2 or 'green' in s2 or 'alaskan' in s2 \
            or 'progressive' in s2 or 'moderate' in s2 or 'constitution' in s2:
        return 'O'
    return None


def summary_table(t):
    for m in re.finditer(r'\{\|[^\n]*wikitable[^\n]*\n(.*?)\n\|\}', t, flags=re.S):
        tb = m.group(0)
        if 'Candidates' in tb[:1500] and re.search(r'Incumbent|Governor', tb[:1500]):
            return tb
    return None


races = []
problems = []
for year in range(1996, 2026):
    fn = os.path.join(D, f'{year}_United_States_gubernatorial_elections.json')
    if not os.path.exists(fn):
        continue
    t = json.load(open(fn))['parse']['wikitext']
    tables = []
    for m in re.finditer(r'\{\|[^\n]*wikitable[^\n]*\n(.*?)\n\|\}', t, flags=re.S):
        tb = m.group(0)
        if 'Candidates' in tb[:2500] and re.search(r'Incumbent|Governor', tb[:2500]) and '{{aye' in tb.lower():
            tables.append(tb)
    if not tables:
        problems.append((year, 'no summary table'))
        continue
    seen = set()
    for tb in tables:
        for row in clean(tb).split('\n|-')[1:]:
            lines = row.split('\n')
            head = next((l for l in lines if l.startswith('!')), None)
            if head is None:
                continue
            st = None
            for s in sorted(STATES, key=len, reverse=True):
                if re.search(r'\b' + re.escape(s) + r'\b', head):
                    st = s
                    break
            if st is None:
                continue
            special = bool(re.search(r'special|recall', head, flags=re.I))
            body = '\n'.join(lines[lines.index(head) + 1:])
            cells = [c.strip() for c in re.split(r'\n\|(?!\})', '\n' + body) if c.strip()]
            # incumbent party: first cell (after the governor name cell) that names a party
            inc = None
            for c in cells[1:3]:
                p = party_word(re.sub(r'\[\[[^\]]*\]\]', '', c)) or party_word(c)
                if p:
                    inc = p
                    break
            result_text = ''
            for c in cells:
                if re.search(r'gain|hold|re-elected|elected|retired|lost|term-limited|recalled|resigned', c, flags=re.I) \
                        and '{{aye' not in c.lower() and 'plainlist' not in c.lower():
                    result_text = re.sub(r'\{\{[^}]*\}\}|\[\[[^|\]]*\||\]\]|\[\[|\'\'\'|<br\s*/?>', ' ', c)
                    result_text = re.sub(r'\s+', ' ', result_text).strip()
            cands = []
            recall_vote = None
            for l in body.split('\n'):
                l2 = l.strip()
                if not l2.startswith('*'):
                    continue
                pct = re.search(r'(\d{1,3}(?:\.\d+)?)\s*%', l2)
                par = re.search(r'\(([^()]*)\)\s*(?:\{\{[^}]*\}\}\s*)?\d{1,3}(?:\.\d+)?\s*%', l2)
                aye = '{{aye' in l2.lower()
                if par and pct:
                    cands.append((party_word(par.group(1)) or 'O', float(pct.group(1)), aye, par.group(1)))
                elif pct and re.search(r"\b(Yes|No)\b", l2):
                    recall_vote = recall_vote or []
                    recall_vote.append((re.search(r"\b(Yes|No)\b", l2).group(1), float(pct.group(1)), aye))
            if not cands:
                problems.append((year, st, 'no candidates'))
                continue
            win = [c for c in cands if c[2]]
            if recall_vote and not win:
                # failed recall: incumbent stays (e.g. California 2021)
                no = [v for v in recall_vote if v[0] == 'No']
                winner = inc
                wpct = no[0][1] if no else None
            elif win:
                winner = win[0][0]
                wpct = win[0][1]
            else:
                problems.append((year, st, 'no winner mark'))
                continue
            rs = [c[1] for c in cands if c[0] == 'R']
            ds = [c[1] for c in cands if c[0] == 'D']
            margin = (max(rs) - max(ds)) if rs and ds else None
            others = sorted((c for c in cands if not c[2]), key=lambda c: -c[1])
            key = (year, st, special)
            if key in seen:
                continue
            seen.add(key)
            races.append({'year': year, 'state': st, 'incumbent_party': inc, 'winner_party': winner,
                          'winner_pct': wpct, 'runnerup_party': others[0][0] if others else None,
                          'runnerup_pct': others[0][1] if others else None, 'margin_rd': margin,
                          'result_text': result_text[:160], 'special': special,
                          'recall': recall_vote is not None})
json.dump(races, open(os.path.join(WORK, 'governor_races.json'), 'w'), indent=0)
from collections import Counter
print('races', len(races), 'by year', sorted(Counter(r['year'] for r in races).items()))
print('problems', problems)
