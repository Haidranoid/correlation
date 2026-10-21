"""Single sequential runner: processes task files from QUEUE (sorted by name), one HTTP request
at a time. Holds an exclusive lock so two runners can never overlap. Exits on STOP file or when
the queue stays empty for IDLE_EXIT seconds.

Task file: JSON list of {"label", "out" (path relative to RAW), "items": [comparisonItem...]}.
"""
import fcntl
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtfetch import Fetcher, RAW, log  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, 'queue')
DONE = os.path.join(HERE, 'done')
STOP = os.path.join(HERE, 'STOP')
IDLE_EXIT = 1800

os.makedirs(QUEUE, exist_ok=True)
os.makedirs(DONE, exist_ok=True)
lock = open(os.path.join(HERE, 'runner.lock'), 'w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print('another runner holds the lock; exiting')
    sys.exit(1)

f = Fetcher(min_gap=13.0, jitter=5.0, tz=300, hl='en-US')
f.cookie()
idle_since = time.time()
failures = {}
while True:
    if os.path.exists(STOP):
        log(ev='stop_file')
        break
    files = sorted(x for x in os.listdir(QUEUE) if x.endswith('.json'))
    if not files:
        if time.time() - idle_since > IDLE_EXIT:
            log(ev='idle_exit')
            break
        time.sleep(5)
        continue
    tf = files[0]
    tasks = json.load(open(os.path.join(QUEUE, tf)))
    log(ev='taskfile_start', file=tf, n=len(tasks))
    for t in tasks:
        if os.path.exists(STOP):
            break
        out = os.path.join(RAW, t['out'])
        if os.path.exists(out):
            continue
        try:
            st = f.fetch_to(t['items'], out, t['label'])
        except Exception as e:  # keep going; record the failure
            failures[t['label']] = str(e)[:300]
            log(ev='task_fail', label=t['label'], err=str(e)[:300],
                tb=traceback.format_exc()[-600:])
            with open(os.path.join(RAW, 'failures.jsonl'), 'a') as ff:
                ff.write(json.dumps({'label': t['label'], 'out': t['out'],
                                     'items': t['items'], 'err': str(e)[:300]}) + '\n')
        log(ev='progress', file=tf, label=t['label'], n_requests=f.n_requests, n_429=f.n_429)
    if os.path.exists(STOP):
        log(ev='stop_file')
        break
    os.replace(os.path.join(QUEUE, tf), os.path.join(DONE, tf))
    log(ev='taskfile_done', file=tf)
    idle_since = time.time()
log(ev='runner_exit', n_requests=f.n_requests, n_429=f.n_429, failures=len(failures))
