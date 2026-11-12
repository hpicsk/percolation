"""Percolation paper replication package.

Submodules:
    constants      - Pipeline-wide constants (window, step, seed, thresholds);
                     the single source of truth imported everywhere else.
    network_utils  - Correlation-network and percolation primitives, the
                     market / sector / own-PCA residualizers, and the
                     signal-eigenvalue count behind the factor batteries.
    windows        - Canonical rolling-window x size-quintile iterator
                     (``iter_window_cells``) shared by every scanning script.
    paths          - Every path the pipeline reads, resolved in one place.
    kr_universe    - The point-in-time Korean common-stock universe, read off
                     the marcap daily snapshots.
    data_loader    - Builds the survivorship-bias-free Korean returns /
                     market-cap panels (Taiwan: see ``tw_data_loader``).
    pipeline       - Rolling percolation pipeline under point-in-time
                     quintile assignment.
    rmt            - Marchenko-Pastur bounds and eigenvalue clipping.
    overlap_inference - Newey-West HAC and moving-block bootstrap for the
                     overlapping-window paired differences.
    sector_loader  - KRX sector snapshots and within-sector quintiles.
    tw_data_loader - The Taiwanese counterpart of data_loader, on FinMind's
                     own 還原股價 and listing registry.
    dgp            - The toy return-generating process and its closed form.
"""
