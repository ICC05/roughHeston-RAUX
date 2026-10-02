# File guide

For the conceptual overview see [`README.md`](README.md).

## Estimation script (entry point)
| File | What it does |
|---|---|
| `estimate_rough_heston_RAUX_new2.m` | Estimates the rough Heston (λ, θ, ν, ρ, H) with RAUX: multistart CMA-ES with adaptive relaunch, then pattern search, then out-of-sample check and noise floor; saves the results. |

## Setup
| File | What it does |
|---|---|
| `setup_data_rough_new.m` | Loads `spy_data.mat`, builds RV and returns, computes the empirical RAUX moments and their bootstrap covariance, pre-tabulates the kernel weights, generates the seeds and starts the parallel pool. |

## Objective function and simulator
| File | What it does |
|---|---|
| `objective_rough_heston_RAUX.m` | Given θ: simulates the replications, computes RAUX on each, returns χ², mean simulated moments, per-moment contributions and t-stats. |
| `simulate_rough_heston.m` | Simulates rough Heston intraday returns (N-factor approximation, burn-in, aggregation to 5 minutes). |

## RAUX auxiliary criterion
| File | What it does |
|---|---|
| `rough_aux_estimate.m` | Computes the 11 RAUX statistics (HAR, variogram, leverage, skewness/kurtosis) from log RV, daily and intraday returns. |
| `rough_aux_bootstrap_vcv.m` | Estimates the covariance of the RAUX moments via moving-block bootstrap (with ridge) → weighting matrix. |
| `HAR_estimate.m` | HAR-RV regression (used inside `rough_aux_estimate`). |
| `LHAR_estimate.m` | LHAR regression (computed in the setup only as a reference/diagnostic). |
| `aggregateAvg.m` | Backward-looking rolling average (weekly/monthly regressors). |
| `nwest.m` | OLS with Newey-West covariance. |

## Fractional kernel
| File | What it does |
|---|---|
| `compute_kernel_weights.m` | Non-negative L²-optimal weights of the sum of exponentials approximating the kernel. |
| `precompute_kernel_table.m` | Tabulates the weights over a grid of H and builds PCHIP interpolants. |
| `interp_kernel_weights.m` | Weights for any H via interpolation. |

## Optimizers
| File | What it does |
|---|---|
| `cmaes_bnd_new.m` | Box-constrained CMA-ES with per-generation relative stopping rule. |
| `pattern_search_bnd_new.m` | Bound-constrained pattern search with Latin Hypercube polls and plateau stopping. |

## Output and logging
| File | What it does |
|---|---|
| `print_estimation_results.m` | Prints and saves the results table (observed vs simulated moments, t-stats, per-moment χ²). |
| `dual_log.m` | Writes messages to screen and to the logfile (used by the estimation script). |
| `dual_log_new.m` | Same, used by CMA-ES and pattern search. |

## Data and results
| File | Content |
|---|---|
| `spy_data.mat` | Cleaned SPY 5-minute returns (days × intervals); the only data file read by the code. |
| `results_rough_heston_RAUX_new2.mat / .txt` | Estimation results. |
| `logfile_rough_heston_RAUX_new2.txt` | Estimation log. |
