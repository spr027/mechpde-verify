"""Unit and Method-of-Manufactured-Solutions (MMS) verification tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mechpde_verify import (
    discrete_heat_residual_1d,
    discrete_poisson_residual_2d,
    grid_convergence_index,
    solve_burgers_1d,
    solve_euler_bernoulli_beam,
    solve_heat_1d,
    solve_poisson_2d,
)
from mechpde_verify.cli import (
    benchmark_burgers_1d,
    benchmark_euler_bernoulli_beam,
    benchmark_heat_1d,
    benchmark_poisson_2d,
    main,
)


def test_heat_1d_ftcs_stability_guard() -> None:
    with pytest.raises(ValueError, match="unstable"):
        solve_heat_1d(
            nx=20,
            nt=10,
            t_final=0.5,
            alpha=1.0,
            u0=lambda x: np.sin(np.pi * x),
            scheme="ftcs",
        )


@pytest.mark.parametrize("scheme", ["ftcs", "backward_euler", "crank_nicolson"])
def test_heat_1d_schemes_match_analytical(scheme: str) -> None:
    alpha = 0.2
    t_final = 0.1
    nx = 32
    nt = 200 if scheme == "ftcs" else 64
    res = solve_heat_1d(
        nx=nx,
        nt=nt,
        t_final=t_final,
        alpha=alpha,
        u0=lambda x: np.sin(np.pi * x),
        scheme=scheme,  # type: ignore[arg-type]
    )
    exact = np.exp(-alpha * (np.pi**2) * t_final) * np.sin(np.pi * res.x)
    assert np.max(np.abs(res.u - exact)) < 5e-3


def test_heat_1d_crank_nicolson_second_order_mms() -> None:
    study = benchmark_heat_1d()
    assert study.asymptotic_order_l2 == pytest.approx(2.0, abs=0.05)
    assert study.asymptotic_order_linf == pytest.approx(2.0, abs=0.05)


def test_poisson_2d_second_order_mms_and_residual() -> None:
    study = benchmark_poisson_2d()
    assert study.asymptotic_order_l2 == pytest.approx(2.0, abs=0.05)
    assert study.asymptotic_order_linf == pytest.approx(2.0, abs=0.05)

    source = lambda x, y: 2.0 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)
    res = solve_poisson_2d(nx=32, ny=32, conductivity=1.0, source=source)
    residual = discrete_poisson_residual_2d(res.u, res.x, res.y, conductivity=1.0, source=source)
    assert np.max(np.abs(residual)) < 1e-11
    assert res.residual_l2 < 1e-11


def test_burgers_1d_second_order_traveling_wave() -> None:
    study = benchmark_burgers_1d()
    assert study.asymptotic_order_l2 == pytest.approx(2.0, abs=0.08)
    assert study.asymptotic_order_linf == pytest.approx(2.0, abs=0.08)


def test_burgers_1d_cfl_guard() -> None:
    with pytest.raises(ValueError, match="diffusion number"):
        solve_burgers_1d(
            nx=50,
            nt=1,
            t_final=1.0,
            nu=0.5,
            u0=lambda x: np.zeros_like(x),
            bc_left=lambda _t: 0.0,
            bc_right=lambda _t: 0.0,
        )


def test_euler_bernoulli_beam_second_order_and_uniform_load() -> None:
    study = benchmark_euler_bernoulli_beam()
    assert study.asymptotic_order_l2 == pytest.approx(2.0, abs=0.05)

    # Uniformly distributed load q0 = 1, L = 1, EI = 1 -> w_max = 5 / 384 at x = 0.5
    res = solve_euler_bernoulli_beam(nx=128, length=1.0, ei=1.0, load=lambda x: np.ones_like(x))
    mid_idx = len(res.x) // 2
    assert res.w[mid_idx] == pytest.approx(5.0 / 384.0, rel=1e-3)


def test_discrete_heat_residual_single_step_zero() -> None:
    x = np.linspace(0.0, 1.0, 33)
    u0 = np.sin(np.pi * x)
    res = solve_heat_1d(nx=32, nt=1, t_final=0.01, alpha=0.5, u0=lambda _x: u0, scheme="crank_nicolson")
    residual = discrete_heat_residual_1d(u0, res.u, res.x, res.dt, alpha=0.5)
    assert np.max(np.abs(residual)) < 1e-11


def test_gci_validation() -> None:
    gci = grid_convergence_index(4e-4, 1e-4, refinement_ratio=2.0, observed_order=2.0)
    assert gci == pytest.approx(125.0, rel=1e-6)
    with pytest.raises(ValueError):
        grid_convergence_index(1e-3, 1e-4, refinement_ratio=1.0, observed_order=2.0)


def test_cli_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "benchmarks.json"
    code = main(["--json-out", str(out_file)])
    assert code == 0
    captured = capsys.readouterr().out
    assert "PASS" in captured
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) == 4
