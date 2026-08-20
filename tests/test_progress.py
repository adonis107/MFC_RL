from __future__ import annotations

import io
import sys

import pytest
import torch

from configs.twostate import TwoStateRunConfig
from mfc.progress import PROGRESS_MODES, progress_enabled, training_bar
from scripts.train import CONFIGS, run_all


class _FakeStderr(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


@pytest.mark.parametrize("mode,tty,expected", [("on", False, True), ("off", True, False), ("auto", True, True), ("auto", False, False)])
def test_progress_enabled_resolves_mode_and_tty(mode, tty, expected, monkeypatch):
    monkeypatch.delitem(sys.modules, "ipykernel", raising=False)
    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty))
    assert progress_enabled(mode) is expected


def test_progress_enabled_auto_draws_a_bar_in_a_notebook_kernel(monkeypatch):
    """A Jupyter kernel's stderr is never a tty, but the notebooks call
    run_all/run_continuous directly and do want a bar, so `isatty` alone
    would wrongly suppress it there."""
    monkeypatch.setattr(sys, "stderr", _FakeStderr(tty=False))
    monkeypatch.setitem(sys.modules, "ipykernel", object())
    assert progress_enabled("auto") is True


def test_progress_enabled_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown progress mode"):
        progress_enabled("yes")
    assert set(PROGRESS_MODES) == {"auto", "on", "off"}


def test_training_bar_without_a_desc_is_a_silent_no_op(capsys):
    """`desc=None` must still be iterable and still accept `set_postfix_str`,
    so a training loop wraps itself unconditionally instead of branching."""
    with training_bar(5, desc=None) as bar:
        seen = []
        for m in bar:
            bar.set_postfix_str(f"J={m}", refresh=False)
            seen.append(m)
    assert seen == [0, 1, 2, 3, 4]
    assert bar.disable is True
    assert capsys.readouterr().err == ""


def test_training_bar_with_a_desc_iterates_the_full_range(capsys):
    with training_bar(4, desc="job") as bar:
        assert bar.disable is False  # checked before iterating: tqdm sets this on close
        assert list(bar) == [0, 1, 2, 3]
    assert "job" in capsys.readouterr().err  # bars go to stderr, leaving stdout to the summary lines


@pytest.mark.parametrize("progress", ["off", "on"])
def test_run_all_prints_one_self_contained_line_per_status(progress, tmp_path, monkeypatch, capsys):
    """Under train_all.sh several workers share one stdout, so a status line
    that only makes sense joined to a later write (the old dangling
    "<tag> ... " prefix) interleaves into ambiguity. Every line must carry
    its own tag."""
    cfg = TwoStateRunConfig(name="test", seeds=(0,), horizons=(2,), lambdas=(0.2,), n_train=3, validate_every=3)
    monkeypatch.setitem(CONFIGS["twostate"], "smoke", cfg)
    kwargs = dict(output_dir=str(tmp_path), progress=progress)

    run_all("twostate", "simplex", "smoke", **kwargs)
    run_all("twostate", "simplex", "smoke", **kwargs)  # second pass: skipped (already exists)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    tag = "simplex_equal_parameters_exact_T2_lam0.2_seed0"
    assert all(ln.startswith(f"[twostate/smoke] {tag} ... ") for ln in lines), lines
    statuses = [ln.split(" ... ", 1)[1] for ln in lines]
    assert any(s.startswith("done in") for s in statuses)
    assert any(s.startswith("skipped") for s in statuses)
    # the start announcement is the bar-less path's only progress signal; with
    # a bar running it would be redundant with the bar itself
    assert ("started" in statuses) is (progress == "off")


def test_train_run_progress_desc_does_not_perturb_the_result(tmp_path, monkeypatch):
    """The bar must be pure display: same seed, same numbers, bar or no bar."""
    cfg = TwoStateRunConfig(name="test", seeds=(0,), horizons=(2,), lambdas=(0.2,), n_train=6, validate_every=2)
    monkeypatch.setitem(CONFIGS["twostate"], "smoke", cfg)

    [off] = run_all("twostate", "simplex", "smoke", output_dir=str(tmp_path / "off"), progress="off")
    [on] = run_all("twostate", "simplex", "smoke", output_dir=str(tmp_path / "on"), progress="on")

    assert torch.equal(off["theta_final"], on["theta_final"])
    assert torch.equal(off["validation_J"], on["validation_J"])
    assert torch.equal(off["theta_history"], on["theta_history"])
