"""Compute fund-like flags (c2eval.twins) on the new export, saved in this experiment's work folder."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import c2eval, twins_for
import time
t = time.time()
E = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('close', 'dollar_volume'))
tw = twins_for('new', E)
print('done', tw['fund_like'].shape, round(time.time() - t), 's')
