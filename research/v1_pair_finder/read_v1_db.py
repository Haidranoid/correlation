"""Read v1's saved relationships (stock_correlations) from its Postgres, read-only, into a TSV.

Connection settings come from correlation-original/.env (PGHOST, PGPORT, PGUSER, PGPASSWORD); they are never printed.
The session is opened read-only (default_transaction_read_only) and only SELECTs are run.

    research/v1_pair_finder/.venv/bin/python -I research/v1_pair_finder/read_v1_db.py
"""
import sys
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
ENV = HERE.parent.parent / 'correlation-original' / '.env'


def env():
    out = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main():
    e = env()
    conn = psycopg.connect(host=e.get('PGHOST', '127.0.0.1'), port=int(e.get('PGPORT', 5434)), user=e.get('PGUSER'),
                           password=e.get('PGPASSWORD'), dbname=sys.argv[1] if len(sys.argv) > 1 else 'correlation_development',
                           options='-c default_transaction_read_only=on', autocommit=False)
    with conn:
        with conn.cursor() as cur:
            cur.execute('SHOW default_transaction_read_only')
            assert cur.fetchone()[0] == 'on'
            cur.execute('SELECT * FROM stock_correlations ORDER BY symbol1, symbol2, window1_days, window2_days, offset_days')
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        conn.rollback()
    out = HERE / 'v1_saved_relationships.tsv'
    with open(out, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f'{len(rows)} rows -> {out}')


if __name__ == '__main__':
    main()
