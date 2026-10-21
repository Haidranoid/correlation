"""The proof of concept's checks and bookkeeping (scripts/poc.sh runs them; test-results.md reports them).

- g0.py: the M0 pipeline check: Python reversal scores against the judge's own, bit for bit, and judge JSON
  outputs field by field (`python -m stockai.poc.g0 ...`).
- controls.py: the score canaries (stale scores shifted later, dates permuted within each year) and the large-cap
  leakage tripwire (`python -m stockai.poc.controls ...`).
- summarize.py: reads every graded JSON and train log, and writes the tables and the ledger of graded
  configurations (`python -m stockai.poc.summarize ...`).

Nothing here reads the locked year (2024-06-28 on): score files are NaN from its first decision day
(stockai.baselines.common.dev_cutoff), and outcomes come only from columns before stockai.gpt.features.locked_cut.
"""
