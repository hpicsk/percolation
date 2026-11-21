# Firm Size and the Percolation Threshold of Stock-Return Correlation Networks

Replication repository for the paper *"Firm Size and the Percolation Threshold of Stock-Return Correlation Networks: Evidence from Korea and Taiwan"* by Sungwoo Kang and Jong-Kook Kim.

## Summary

Correlation-based networks of equity returns undergo a percolation transition at a threshold $\rho_c$. Prior work reports $\rho_c$ as one number per market, estimated on large, liquid samples. This paper estimates $\rho_c$ in each size quintile of two thin markets. Each window's $\rho_c$ is averaged over 20 draws of each quintile's stocks. On the survivorship-bias-free Korea Exchange (KOSPI and KOSDAQ) common-stock universe and on a Taiwan (TWSE and TPEx) common-stock panel, the mean $\rho_c$ rises at each step from the smallest to the fourth quintile, and autocorrelation-robust inference finds each of these steps significant but not the step to the largest quintile. Weekly returns reproduce each Taiwanese step through the fourth quintile, whereas in Korea they reproduce at both of two window lengths only the first step and the endpoint contrast. A return-generating model predicts the sign of the gradient if a segment-level local-absorption fraction $\delta$ falls with firm size, a premise that the paper does not measure. The gradient is a property of total-return correlation, and in Korea it compresses over the second half of the sample.

## Contents

```
paper/                          # Manuscript + Supplementary Material (LaTeX source, .bib, compiled PDFs)
figures/                        # Compiled figure PDFs
data/                           # Per-stage analysis outputs (one subdir per analysis)
  cache/                        #   Built panels (returns, marcap)
  korea_headline/               #   Primary: point-in-time SBF rolling rho_c (Korea)
  taiwan/                       #   Cross-market re-run (TWSE + TPEx)
  susceptibility/               #   Susceptibility-peak rho_c (textbook observable)
  rmt_filtered/                 #   Marchenko-Pastur eigenvalue-clipped rho_c
  null_shuffled/                #   Shuffled-returns null
  subperiod/                    #   Subperiod attenuation (5-year calendar blocks)
  weekly_rho_c/                 #   rho_c from weekly returns on 252- and 504-day windows
  bandwidth_sensitivity/        #   Headline tests at 12- and 24-window bandwidths
  window_sensitivity/           #   Window-length sensitivity (30/60/120/252-day)
  min_valid_fraction_sensitivity/  #   Observation-requirement sensitivity (0.7/0.8/0.9)
  network_size_resampling/      #   rho_c re-sampled over the network size N
  kospi_only/                   #   Venue confound control
  within_sector/                #   Sector-mix confound control
  mean_corr_diagnostic/         #   rho_c vs mean |rho_ij| co-movement
  dgp_validation/               #   DGP simulation of the toy model
src/fn_percolation/             # Inlined pipeline library
scripts/                        # Analysis stages, one folder per family (see reproduce.sh)
  build_panels.py               #   Step 0: caches every input panel, KR + TW
  headline/                     #   Point-in-time pipeline on both markets, chi-peak
                                #   observable, random 5-year windows, claim verification
  confounds/                    #   Venue and sector-mix (with its floor sweep)
  nulls/                        #   Shuffled-returns and RMT-filtered null models
  factor_residual/              #   External-factor residual arms (one script, four
                                #   arms) and the per-quintile return kurtosis
  scaling/                      #   Network-size re-sampling and rho_c-vs-mean|rho|
  stability/                    #   Subperiod split and rho_c from weekly returns
  sensitivity/                  #   Window length and observation requirement
  dgp/                          #   Toy-model simulation
  generate_params.py            #   Last step: writes paper/params.tex
generate_figures.py             # Reproduces figures from bundled data
test_assertions.py              # Pytest: every paper number checked against data
reproduce.sh                    # Single entry-point: runs the full pipeline
                                # (reproduce.sh also writes reports/verification_report.md,
                                #  an internal working document, untracked)
requirements.txt                # Python dependencies
```

## Data

Every dataset the pipeline reads sits under `data/external/` in a fixed layout
that `src/fn_percolation/paths.py` documents and resolves; `FNP_DATA_ROOT`
relocates the whole root. Nothing is read from outside this repository.

The Korean panel is **survivorship-bias-free**. It is built from the
open-source `marcap` archive, a daily snapshot of every code KRX quoted that
session, which keeps a code after its delisting; the universe is the
point-in-time common stocks of KOSPI and KOSDAQ, read off those snapshots by
`src/fn_percolation/kr_universe.py`, so pre-listing and post-delisting cells
stay NaN. Korean returns come from the FnGuide DataGuide 수정주가 export, a
paid subscription whose raw files are not redistributed here. The universe
admits only the sessions that export prices.

The Taiwanese panel is built from FinMind's own series — `TaiwanStockPriceAdj`
(還原股價) for returns, `TaiwanStockPrice` for the traded print, and
`TaiwanStockInfo` for the universe. The price endpoints also serve the sessions
on which a stock traded on the TPEx emerging board (興櫃). FinMind's share
counts (`shares/`) mark the listed sessions: a session enters the panel only if
FinMind reports the stock's share count on it. FinMind's live listing registry
drops some of the companies the exchange has delisted, and FinMind's
adjusted-price endpoint serves few of the names the registry drops, so the
Taiwanese panel is not survivorship-bias-free;
`tw_data_loader.delisted_in_window` measures that exposure against FinMind's
own delisting record and `src/fn_percolation/tw_data_loader.py` states what the
vendor does and does not answer for.

### Data requirements

| Path under `data/external/` | Supplied by | Committed |
|---|---|---|
| `marcap/` | `git clone https://github.com/FinanceData/marcap` | no |
| `fnguide/` | FnGuide DataGuide 수정주가 export (subscription) | no |
| `finmind/{ohlcv,price_adj,shares}/` | FinMind download (free tier, token) | no |
| `registry/` | — | yes |
| `krx_supplement/` | — | yes |

A missing tree raises `RuntimeError` naming it and what supplies it, so a
replicator learns which one to place rather than reading a stack trace. Only
`scripts/build_panels.py` touches any of them: the panels it writes are
committed under `data/cache/`, so every published table and figure reproduces
with all three absent.

The bundled CSVs under `data/korea_headline/` are derived results — rolling percolation
thresholds at 60-day windows stepped 20 trading days apart, with quintile
membership reassigned point-in-time at every window.

### What is committed under `data/`

`data/cache/` holds the built input panels. They are the only artifacts
`reproduce.sh` cannot rebuild from this repository alone, because step 0 reads
the marcap, FnGuide, and FinMind trees, so they are committed in full.

Everything else under `data/` is committed selectively: an artifact stays
tracked when `test_assertions.py`, `scripts/generate_params.py`, or
`generate_figures.py` opens it, so a fresh clone can run the assertion suite,
regenerate `paper/params.tex`, and rebuild every figure without re-running the
pipeline. Per-seed checkpoints, per-cell intermediates, and per-quintile files
that only feed a later stage are `.gitignore`d — `reproduce.sh` writes them
from `data/cache/` on a full run. The `.gitignore` block naming them records
how to re-derive the list after a pipeline change.

## Reproduction

```bash
pip install -r requirements.txt
pip install -e .          # editable install of the in-repo fn_percolation package

# Everything else: one shell script, about four hours on twenty or more cores
bash reproduce.sh
```

`reproduce.sh` runs all caches, every pipeline stage, the headline-claim
verifier, the manuscript-claim assertions (`pytest test_assertions.py`),
the figure generator, and `latexmk`, in order. Each stage writes to a
deterministic output directory under `data/` and is idempotent.

## Building the paper

```bash
cd paper && for _ in 1 2 3; do latexmk -pdf paper.tex; latexmk -pdf paper_si.tex; done
```

The two documents cross-reference each other through `xr-hyper`, so each needs
the other's `.aux` file and one pass is not enough on a clean checkout: a single
round leaves the supplement's references to the main paper unresolved. Three
alternating rounds reach a fixed point with no undefined reference in either
document.

Requires a standard TeX Live distribution plus the Elsevier CAS bundle (`cas-sc` class, vendored
under `els-cas-templates/`), with `natbib`, `booktabs`, `threeparttable`,
`float`, `xr-hyper`, and `hyperref`.

## License

- Source code and data (`generate_figures.py`, `src/`, `scripts/`, `data/`): [MIT License](LICENSE)
- Manuscript content (`paper/`, `figures/`): [CC-BY-4.0](paper/LICENSE-PAPERS)

## Citation

```
Kang, S., Kim, J.-K. (2026). Firm Size and the Percolation Threshold of
Stock-Return Correlation Networks: Evidence from Korea and Taiwan.
```

## Contact

Jong-Kook Kim (corresponding author) — jongkook@korea.ac.kr
School of Electrical Engineering, Korea University, Seoul

Sungwoo Kang — krml919@korea.ac.kr
Department of Electrical and Computer Engineering, Korea University, Seoul
