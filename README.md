# Rough Heston — indirect inference estimation with the RAUX auxiliary criterion

MATLAB code for the **indirect inference** (EMSM) estimation of the **rough Heston** model on 5-minute returns of the SPDR S&P 500 ETF (SPY). The auxiliary criterion is **RAUX**, a purpose-built vector of 11 statistics capturing the persistence, roughness, leverage and tail shape of realized volatility.

The code accompanies the research project *Indirect inference estimation of the rough Heston model* (I. Cardosi Carrara, Sant'Anna School of Advanced Studies, 2026), Sections 7.1 and 7.3 of the project report.

---

## 1. Why this repository

Estimated against the LHAR criterion (see the [Heston-doubleHeston-roughHeston-LHAR](https://github.com/ICC05/Heston-doubleHeston-roughHeston-LHAR) repository), the rough Heston fails: χ² = 701, with the wrong leverage sign (ρ̂ = +0.38). The problem lies in the criterion, not the model:
- the LHAR contains **no statistic sensitive to roughness**, i.e. to the small-scale scaling of log-volatility governed by H. H is therefore left undetermined and confounded with the mean-reversion speed λ;
- the LHAR leverage regressors deliver a **nearly flat binding function in ρ**.

To give the rough model a fair metric, the project introduces the **RAUX** criterion (original to this work).

## 2. The idea: indirect inference

1. Compute the auxiliary statistics on the **observed data** → β̂.
2. For each candidate θ, **simulate** many model paths and compute the same statistics with the same routine; the average over replications is the simulated *binding function* b̂(θ).
3. Minimize χ²(θ) = [β̂ − b̂(θ)]ᵀ Ω [β̂ − b̂(θ)], where Ω is the inverse covariance of the empirical moments.

The random shocks are held fixed across candidate θ (common random numbers), which keeps the objective surface smooth.

## 3. Data

SPY, 1990–2013, cleaned 5-minute returns: low-activity days removed, outliers discarded after local standardization, August–December 2008 excluded. The final sample has **T = 5,867 days** (`spy_data.mat`). Daily realized variance is the sum of squared intraday returns, excluding overnight.

## 4. Structural model: rough Heston

dSₜ/Sₜ = μ dt + √Vₜ dWₜ

Vₜ = V₀ + (1/Γ(α)) ∫₀ᵗ (t−s)^(α−1) λ(θ − Vₛ) ds + (1/Γ(α)) ∫₀ᵗ (t−s)^(α−1) λν √Vₛ dBₛ,  dWₜ dBₜ = ρ dt

with H = α − ½ ∈ (0, ½). The parameters are **θ = (λ, θ, ν, ρ, H)**, so p = 5. For H = ½ the model reduces to the classical Heston.

### Simulation
- **Multifactor Markovian approximation** (Abi Jaber and El Euch, 2019): the singular kernel (t−s)^(α−1)/Γ(α) is replaced by a sum of **N = 20 exponentials**. Nodes lie on a geometric grid between 0.1 and 100 (1/day); weights come from non-negative L² least squares on [0, 44] days.
- The weights are **pre-tabulated** on H ∈ {0.02, 0.025, …, 0.49} and interpolated with PCHIP, so the binding function stays smooth in H.
- **Stable grid**: M = 400 steps per day (γ_max·Δt = 0.25 < 1). Returns are then aggregated to 80 five-minute intervals, as in the data.
- **Burn-in** of 1,000 days (the process is non-Markovian), followed by 6,000 days used for the statistics.
- Full-truncation correction for variance positivity.

### Variance reduction
- **H = 494 effective replications**: 247 independent streams plus their antithetic versions.
- Shocks are not stored but regenerated from 247 fixed Threefry seeds.
- Replications are computed in parallel.

## 5. RAUX auxiliary criterion (q = 11)

Unlike the LHAR, RAUX is **not a model**. It is a vector of statistics computed with the same routine on the data and on the simulations:

| Block | Statistic | Informs |
|---|---|---|
| Level and persistence (HAR-RV) | mean of log RV | θ |
| | HAR-RV β⁽ᵈ⁾, β⁽ʷ⁾, β⁽ᵐ⁾ | λ (persistence) |
| | HAR residual variance | ν |
| **Roughness** (variogram) | slope b_vg and intercept a_vg of the regression of log m₁(τ) on log τ, τ ∈ {2, 5, 10, 22}, where m₁(τ) = mean of \|log RVₜ₊τ − log RVₜ\| | H (b_vg), ν (a_vg) |
| **Leverage** | corr(rₜ, log RVₜ₊₁) | ρ |
| | corr((RS⁻ − RS⁺)/RV, log RVₜ₊₁), from realized semivariances (Patton and Sheppard, 2015) | ρ |
| **Tail shape** | skewness of daily returns | ρ |
| | excess kurtosis | ν |

With 11 moments and 5 parameters the system is **over-identified** with 6 restrictions, so the minimized χ² is a genuine goodness-of-fit test.

**Weighting matrix.** Most RAUX moments are not OLS coefficients, so there is no closed-form Newey-West covariance. The covariance is estimated by a **moving-block bootstrap**: B = 1,000 resamples in 44-day blocks, drawn jointly across log RV, daily and intraday returns. A small ridge term guarantees invertibility.

## 6. Optimization and diagnostics

1. **CMA-ES**: population 20, up to 300 generations, 10 restarts with Latin Hypercube initialization. Restarts with χ² ≥ 5× the best are relaunched from an antithetic point (at most 3).
2. **Bound-constrained pattern search**, from 5 starting points around the optimum.

Bounds: λ ∈ [0.001, 50], θ ∈ [0.01, 5], ν ∈ [0.01, 25], ρ ∈ [−0.999, 0.999], H ∈ [0.02, 0.49].

Diagnostics:
- **out-of-sample** χ² under independent seeds;
- Monte Carlo **noise floor** over 10 seeds, with signal-to-noise ratio;
- **cross-restart dispersion of Ĥ** (raw and filtered);
- **per-moment** χ² decomposition.

## 7. Results (from the project report)

| | Estimate |
|---|---|
| λ | 0.0271 |
| θ | 0.4928 |
| ν | 13.538 |
| ρ | −0.3606 |
| H | 0.0200 |
| In-sample χ² | **483.87** |
| Out-of-sample χ² | 483.49 |

- **Improvement over LHAR**: χ² falls from 701 to 484, and the leverage effect is recovered with the **correct sign** (ρ̂ < 0). Ĥ is stable across well-converged restarts (it ends at the lower bound 0.02).
- **The model is still rejected**:
  - persistence concentrated at the daily horizon (simulated β⁽ᵐ⁾ ≈ −0.01 vs 0.22);
  - variogram too steep (0.19 vs 0.12);
  - kurtosis too low (2.0 vs 4.6) and skewness too negative.
- On the **same RAUX metric** the double Heston attains χ² = 11.84, about 40 times better (see the [doubleHeston-RAUX](https://github.com/ICC05/doubleHeston-RAUX) repository).

**Conclusion**: a single rough factor, even when judged by a criterion built for it, cannot jointly reproduce the persistence, leverage and tails of realized volatility.

These conclusions concern the **reproduction** of the moments, not the **identification** of the individual parameters, which is left for future work.

## 8. How to run

Requirements: base MATLAB plus Parallel Computing Toolbox (recommended).

From the repository folder, run in MATLAB:

```matlab
estimate_rough_heston_RAUX_new2
```

Output in the current folder:
- `results_rough_heston_RAUX_new2.mat`: struct `est` with parameters, χ², simulated moments and diagnostics, plus `coeffs_aux`, `VCV_aux` and `kernel_table`;
- `results_rough_heston_RAUX_new2.txt`: results table;
- `logfile_rough_heston_RAUX_new2.txt`: full log.

The number of replications (`cfg.n_indep = 247` in `setup_data_rough_new.m`) is tuned for a 247-worker pool. On a standard PC it is worth lowering it, keeping in mind that Monte Carlo noise increases.

A description of every file is in [`CODE.md`](CODE.md).

## Main references
- Abi Jaber, E. & El Euch, O. (2019). *Multifactor approximation of rough volatility models*. SIAM J. Financial Math.
- El Euch, O. & Rosenbaum, M. (2019). *Characteristic function of rough Heston models*. Math. Finance.
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). *Volatility is rough*. Quant. Finance.
- Patton, A. J. & Sheppard, K. (2015). *Good volatility, bad volatility*. REStat.
- Corsi, F. & Renò, R. (2012). JBES.
