"""Random-matrix-theory filtering of empirical correlation matrices.

A T x N sample of independent N(0,1) variables produces a sample correlation
matrix whose eigenvalues, in the limit T, N -> infty with Q = T/N fixed, are
distributed within the Marchenko-Pastur band

    lambda_+/- = sigma^2 (1 +/- sqrt(N/T))^2.

Eigenvalues outside this band are not consistent with noise and carry signal;
eigenvalues inside the band are statistically indistinguishable from noise.
The "eigenvalue clipping" filter (Laloux et al. 1999) replaces all bulk
eigenvalues by their mean (preserving the trace) and reconstructs the
correlation matrix from the modified spectrum. The leading "market" mode is
always above lambda_+ in practice and is left untouched.

We use this filter to test whether the cross-quintile rho_c gradient documented
in the headline analysis survives removal of the noise spectrum. A surviving
gradient is signal; a vanishing one is consistent with the gradient being a
finite-sample / effective-dimensionality artifact.

References:
- Laloux, Cizeau, Bouchaud, Potters (1999), "Noise dressing of financial
  correlation matrices", Phys. Rev. Lett. 83.
- Plerou, Gopikrishnan, Rosenow, Amaral, Guhr, Stanley (2002), "Random matrix
  approach to cross correlations in financial data", Phys. Rev. E 65.
"""

from __future__ import annotations

import numpy as np

# Floor on the diagonal of the cleaned correlation matrix before square-root
# normalisation, to guard against numerical zero/negative values from the
# eigenvalue clipping reconstruction.
DIAG_CLIP_FLOOR = 1e-30


def marchenko_pastur_bounds(T: int, N: int, sigma2: float = 1.0) -> tuple[float, float]:
    """Marchenko-Pastur support [lambda_-, lambda_+] for the sample correlation
    matrix of a T x N i.i.d. block with variance sigma2.

    Valid for both T >= N and T < N (the rank-deficient case): the formula
    describes the support of the *nonzero* eigenvalues of the sample covariance.
    """
    if T <= 0 or N <= 0:
        raise ValueError(f"T and N must be positive (got T={T}, N={N})")
    q = T / N
    lam_minus = sigma2 * (1.0 - np.sqrt(1.0 / q)) ** 2
    lam_plus = sigma2 * (1.0 + np.sqrt(1.0 / q)) ** 2
    return float(lam_minus), float(lam_plus)


def rmt_filter(corr: np.ndarray, T: int) -> np.ndarray:
    """Eigenvalue-clipping RMT filter (Laloux et al. 1999).

    Eigenvalues of the sample correlation matrix falling inside the
    Marchenko-Pastur noise band [lambda_-, lambda_+] are replaced by their
    average, preserving the trace. Eigenvalues above lambda_+ are kept as
    signal (including the leading market mode). The diagonal of the cleaned
    matrix is forced to one to guarantee a valid correlation matrix.

    Parameters
    ----------
    corr : np.ndarray
        Symmetric NxN sample correlation matrix (diagonal = 1 expected).
    T : int
        Number of time-series observations used to build the correlation
        (here, the rolling window length, T=60 in the headline pipeline).

    Returns
    -------
    np.ndarray
        Cleaned NxN correlation matrix.
    """
    N = corr.shape[0]
    if corr.shape != (N, N):
        raise ValueError(f"corr must be square, got shape {corr.shape}")

    # Symmetrize defensively (small floating-point asymmetries arise after
    # NaN handling) then eigendecompose.
    sym = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    lam_minus, lam_plus = marchenko_pastur_bounds(T, N)

    bulk_mask = (eigvals >= lam_minus) & (eigvals <= lam_plus)
    eigvals_clean = eigvals.copy()
    if bulk_mask.any():
        bulk_mean = float(eigvals[bulk_mask].mean())
        eigvals_clean[bulk_mask] = bulk_mean

    cleaned = (eigvecs * eigvals_clean) @ eigvecs.T
    # Re-normalize diagonal: enforce C_ii = 1
    d = np.sqrt(np.clip(np.diag(cleaned), DIAG_CLIP_FLOOR, None))
    cleaned = cleaned / d[:, None] / d[None, :]
    np.fill_diagonal(cleaned, 1.0)
    return cleaned
