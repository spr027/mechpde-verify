"""Canonical computational mechanics MMS benchmark suite and CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mechpde_verify.solvers.beam1d import solve_euler_bernoulli_beam
from mechpde_verify.solvers.burgers1d import (
    solve_burgers_1d,
    viscous_burgers_traveling_wave,
)
from mechpde_verify.solvers.heat1d import solve_heat_1d
from mechpde_verify.solvers.poisson2d import solve_poisson_2d
from mechpde_verify.verification.mms import (
    ConvergenceStudy,
    ErrorMetrics,
    compute_error_metrics,
    run_convergence_study,
)


def benchmark_heat_1d() -> ConvergenceStudy:
    """1D transient heat conduction MMS benchmark (Crank-Nicolson, O(h^2 + dt^2))."""
    alpha = 0.25
    t_final = 0.2

    def exact(x: np.ndarray, t: float) -> np.ndarray:
        return np.exp(-alpha * (np.pi**2) * t) * np.sin(np.pi * x)

    def eval_grid(nx: int) -> ErrorMetrics:
        res = solve_heat_1d(
            nx=nx,
            nt=nx,
            length=1.0,
            t_final=t_final,
            alpha=alpha,
            u0=lambda x: exact(x, 0.0),
            scheme="crank_nicolson",
        )
        return compute_error_metrics(
            res.u,
            exact(res.x, t_final),
            cell_measure=res.dx,
            h=res.dx,
        )

    return run_convergence_study(
        name="1D Transient Heat (Crank-Nicolson)",
        grid_sizes=(16, 32, 64, 128),
        evaluator=eval_grid,
        formal_order=2.0,
    )


def benchmark_poisson_2d() -> ConvergenceStudy:
    """2D steady-state heat conduction / Poisson MMS benchmark (5-point FD, O(h^2))."""
    k = 1.5

    def exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2.0 * k * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def eval_grid(n: int) -> ErrorMetrics:
        res = solve_poisson_2d(nx=n, ny=n, conductivity=k, source=source)
        xx, yy = np.meshgrid(res.x, res.y, indexing="ij")
        return compute_error_metrics(
            res.u,
            exact(xx, yy),
            cell_measure=res.dx * res.dy,
            h=res.dx,
        )

    return run_convergence_study(
        name="2D Steady Poisson (Sparse 5-Point FD)",
        grid_sizes=(16, 32, 64, 128),
        evaluator=eval_grid,
        formal_order=2.0,
    )


def benchmark_burgers_1d() -> ConvergenceStudy:
    """1D viscous Burgers traveling-wave benchmark (Conservative FV + RK2, O(h^2))."""
    nu = 0.05
    t_final = 0.15

    def exact(x: np.ndarray, t: float) -> np.ndarray:
        return viscous_burgers_traveling_wave(x, t, nu=nu, x0=0.3, c=0.5)

    def eval_grid(nx: int) -> ErrorMetrics:
        dx = 1.0 / nx
        # Keep nu * dt / dx^2 <= 0.25 for RK2 stability while refining dt <= O(dx^2)
        nt = max(32, int(np.ceil(t_final * nu / (0.2 * dx * dx))))
        res = solve_burgers_1d(
            nx=nx,
            nt=nt,
            length=1.0,
            t_final=t_final,
            nu=nu,
            u0=lambda x: exact(x, 0.0),
            bc_left=lambda t: float(exact(np.array([0.0]), t)[0]),
            bc_right=lambda t: float(exact(np.array([1.0]), t)[0]),
        )
        return compute_error_metrics(
            res.u,
            exact(res.x, t_final),
            cell_measure=res.dx,
            h=res.dx,
        )

    return run_convergence_study(
        name="1D Viscous Burgers (Conservative FV + RK2)",
        grid_sizes=(32, 64, 128, 256),
        evaluator=eval_grid,
        formal_order=2.0,
    )


def benchmark_euler_bernoulli_beam() -> ConvergenceStudy:
    """1D Euler-Bernoulli simply supported beam under sinusoidal load (O(h^2))."""
    ei = 2.0

    def load(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x)

    def exact_w(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) / (ei * (np.pi**4))

    def eval_grid(nx: int) -> ErrorMetrics:
        res = solve_euler_bernoulli_beam(nx=nx, length=1.0, ei=ei, load=load)
        return compute_error_metrics(
            res.w,
            exact_w(res.x),
            cell_measure=res.dx,
            h=res.dx,
        )

    return run_convergence_study(
        name="1D Euler-Bernoulli Beam (Coupled 2nd-Order FD)",
        grid_sizes=(16, 32, 64, 128),
        evaluator=eval_grid,
        formal_order=2.0,
    )


def run_all_benchmarks() -> list[ConvergenceStudy]:
    """Execute all four canonical computational mechanics benchmarks."""
    return [
        benchmark_heat_1d(),
        benchmark_poisson_2d(),
        benchmark_burgers_1d(),
        benchmark_euler_bernoulli_beam(),
    ]


def format_markdown_summary(studies: list[ConvergenceStudy]) -> str:
    """Format convergence studies as a Markdown summary table."""
    lines = [
        "| Benchmark | Finest Mesh `h` | Finest `L2` Error | Finest `Linf` Error | Observed Order (`L2`) | Formal Order | Status |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for s in studies:
        finest = s.metrics[-1]
        ok = abs(s.asymptotic_order_l2 - s.formal_order) < 0.15
        status = "PASS" if ok else "CHECK"
        lines.append(
            f"| {s.name} | `{finest.h:.5f}` | `{finest.l2:.3e}` | `{finest.linf:.3e}` "
            f"| `{s.asymptotic_order_l2:.3f}` | `{s.formal_order:.1f}` | {status} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for mechpde-verify."""
    parser = argparse.ArgumentParser(
        prog="mechpde-verify",
        description="Run reproducible computational mechanics PDE verification benchmarks.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full benchmark metrics as JSON.",
    )
    args = parser.parse_args(argv)

    studies = run_all_benchmarks()
    print(format_markdown_summary(studies))

    if args.json_out is not None:
        payload: list[dict[str, Any]] = [asdict(s) for s in studies]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
