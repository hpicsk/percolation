#!/usr/bin/env python3
"""Data-generating-process validation for paper si:dgp_validation.

Simulates the toy model of paper eq:toy_rgp and feeds simulated returns through
the same percolation pipeline used for the empirical analysis (
network_utils.percolation_and_susceptibility_scan + find_percolation_threshold),
so the comparison between sim and empirical rho_c is byte-identical at the
identification step.

One experiment checks the approximation stack behind eq:toy_closed_form:

  Exp 1 (criterion) -- Verify that the percolation rho_c matches the
        random-graph degree-one criterion of paper eq:toy_threshold
        directly, rho_c = F_delta^{-1}(1 - 1/(N-1)). Side diagnostic: how far
        the Gaussian Mills-ratio approximation of eq:toy_closed_form
        sits from rho_c.

Two peer-coupling-matrix variants are run: random k-link (k=5) and sector-block
(10 sectors, intra=1.0, inter=0.1).

Outputs:
  data/dgp_validation/eq9_verification.csv

Run:
  python scripts/dgp/run_dgp_validation.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from fn_percolation.constants import PER_QUINTILE, SEED_BASE, WINDOW
from fn_percolation.dgp import (
    DEFAULT_SIGMA_F,
    DEFAULT_SIGMA_U,
    draw_betas,
    make_peer_matrix,
    percolation_one_sim,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "dgp_validation"

# Pipeline-fixed simulation constants, anchored to constants.py so simulated
# rho_c is identified under byte-identical machinery as the empirical pipeline
# (SEED_BASE is imported directly from the same source).
T_WINDOW = WINDOW
N_DEFAULT = PER_QUINTILE
W_TYPES = ("knn", "sector")

# Experiment-1 delta grid.
DELTA_GRID_EXP1 = (0.40, 0.50, 0.60, 0.70, 0.85)

# MC counts locked to the published configuration (si:dgp_validation).
DEFAULT_MC = {
    "exp1": 100,  # 5 delta * 2 W = 10 cells * 100 reps
}


def _seed(exp_id: int, mc_rep: int) -> np.random.Generator:
    """Deterministic seed convention: SEED_BASE + exp_id * 1000 + mc_rep."""
    return np.random.default_rng(SEED_BASE + exp_id * 1000 + mc_rep)


# ---------------------------------------------------------------------------
# Exp 1: closed-form / criterion verification.
# ---------------------------------------------------------------------------

def run_exp1(n_mc: int) -> pd.DataFrame:
    """At each (delta, W), MC-average rho_c_sim, rho_c_degree_one, and
    rho_c_closed across n_mc replicates with independent W and beta draws."""
    print(f"[Exp 1] criterion / closed-form verification: 5 delta x 2 W x {n_mc} reps")
    rows: list[dict] = []
    for W_type in W_TYPES:
        for delta in DELTA_GRID_EXP1:
            t0 = time.time()
            rho_sim, rho_d1, rho_closed = [], [], []
            mu_b, sig_b = [], []
            for rep in range(n_mc):
                rng = _seed(1, rep + 100 * int(delta * 100)
                            + 1_000_000 * (W_type == "sector"))
                W = make_peer_matrix(W_type, N_DEFAULT, rng)
                beta = draw_betas(N_DEFAULT, rng)
                r = percolation_one_sim(
                    N_DEFAULT, T_WINDOW, float(delta), W, beta,
                    DEFAULT_SIGMA_F, DEFAULT_SIGMA_U, rng,
                )
                rho_sim.append(r["rho_c"])
                rho_d1.append(r["rho_c_degree_one"])
                rho_closed.append(r["rho_c_closed"])
                mu_b.append(r["mu_bulk"])
                sig_b.append(r["sigma_bulk"])
            rows.append({
                "delta": float(delta),
                "W_type": W_type,
                "n_reps": n_mc,
                "rho_c_sim_mean": float(np.mean(rho_sim)),
                "rho_c_sim_se": float(np.std(rho_sim, ddof=1)
                                       / np.sqrt(n_mc)),
                "rho_c_degree_one_mean": float(np.mean(rho_d1)),
                "rho_c_degree_one_se": float(np.std(rho_d1, ddof=1)
                                              / np.sqrt(n_mc)),
                "rho_c_closed_mean": float(np.mean(rho_closed)),
                "mu_bulk_mean": float(np.mean(mu_b)),
                "sigma_bulk_mean": float(np.mean(sig_b)),
            })
            print(f"  [{W_type:6s}] delta={delta:.2f}: "
                  f"rho_c_sim={rows[-1]['rho_c_sim_mean']:.4f}, "
                  f"rho_c_deg1={rows[-1]['rho_c_degree_one_mean']:.4f}, "
                  f"rho_c_closed={rows[-1]['rho_c_closed_mean']:.4f}, "
                  f"({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    df = run_exp1(DEFAULT_MC["exp1"])
    df.to_csv(OUT_DIR / "eq9_verification.csv", index=False)
    print(f"  wrote {OUT_DIR / 'eq9_verification.csv'}")

    print(f"\nTotal: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
