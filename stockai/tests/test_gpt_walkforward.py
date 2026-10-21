import json
import os

import numpy as np
import pytest
import torch

from stockai import data, splits
from stockai.gpt import features
from stockai.gpt.walkforward import main, parse_years
from stockai.synthetic import make_synthetic_export

TINY = ["--layers", "1", "--width", "32", "--heads", "2", "--batch", "16", "--eval-batch", "512"]


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    return make_synthetic_export(tmp_path_factory.mktemp("wf"), n=40, seed=5)


def test_parse_years():
    assert parse_years(["2010-2013"]) == [2010, 2011, 2012, 2013]
    assert parse_years(["2015"]) == [2015]
    assert parse_years(["2012,2010", "2011"]) == [2010, 2011, 2012]
    with pytest.raises(ValueError):
        parse_years(["2013-2010"])


def test_quick(export_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--export", str(export_dir), "--out", str(out), "--quick", "--steps", "20", "--eval-every", "10",
                 "--seeds", "3", "4", *TINY]) == 0
    ex = data.load(export_dir)
    s = np.load(out / "scores_seed3.npy")
    assert s.shape == (ex.n, ex.t) and s.dtype == np.float32
    assert not (out / "scores_seed4.npy").exists()  # --quick: the first seed only
    np.testing.assert_array_equal(np.load(out / "scores_ensemble.npy"), s)
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [x["year"] for x in lines] == [2022, 2023] and all(x["seed"] == 3 for x in lines)
    for key in ("split", "steps", "best_step", "val_ce", "bits_gain", "val_ic_actual", "val_ic_v", "train_seconds",
                "params"):
        assert all(key in x for x in lines), key
    expected = np.zeros(s.shape, dtype=bool)
    for y in (2022, 2023):
        test = splits.split(ex.days, y).test_decisions
        expected[:, test.start:test.stop] = np.isfinite(ex.v[:, test.start:test.stop])
        assert (out / "checkpoints" / f"seed3_year{y}.pt").exists()
    assert np.array_equal(np.isfinite(s), expected)  # scored exactly where v[d] is finite in the test years
    assert not np.isfinite(s[:, features.locked_cut(ex.days):]).any()


def test_two_seeds_ensemble_and_null(export_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--export", str(export_dir), "--out", str(out), "--years", "2015", "--seeds", "0", "1",
                 "--steps", "10", "--null", "sign-flip", *TINY]) == 0
    a, b = np.load(out / "scores_seed0.npy"), np.load(out / "scores_seed1.npy")
    assert not np.array_equal(a, b, equal_nan=True)
    np.testing.assert_allclose(np.load(out / "scores_ensemble.npy"), (a + b) / 2, rtol=1e-6, equal_nan=True)
    run = json.loads((out / "run.json").read_text())
    assert run["null"] == "sign-flip" and run["null_seed"] == 0


def test_refuses_to_mix_runs(export_dir, tmp_path):
    out = tmp_path / "out"
    args = ["--export", str(export_dir), "--out", str(out), "--years", "2015", "--seeds", "0", "--steps", "5", *TINY]
    assert main(args) == 0
    with pytest.raises(SystemExit):
        main(args)
    assert main([*args, "--overwrite"]) == 0
    assert len((out / "train_log.jsonl").read_text().splitlines()) == 1


def test_refuses_the_locked_year(export_dir, tmp_path):
    with pytest.raises(ValueError, match="locked"):
        main(["--export", str(export_dir), "--out", str(tmp_path / "out"), "--years", "2023-2024", *TINY])


@pytest.mark.skipif(not torch.cuda.is_available() or bool(os.environ.get("STOCKAI_SKIP_COMPILE")),
                    reason="needs CUDA (STOCKAI_SKIP_COMPILE=1 skips it)")
def test_compile_check(export_dir, tmp_path):
    """--compile runs only after compiled and eager outputs agree (pytorch#191433)."""
    out = tmp_path / "out"
    assert main(["--export", str(export_dir), "--out", str(out), "--years", "2015", "--seeds", "0", "--steps", "10",
                 "--compile", *TINY]) == 0
    line = json.loads((out / "train_log.jsonl").read_text().splitlines()[0])
    assert max(line["compile_check"].values()) <= 1e-2
