function [y, betamean, stdbeta, chi2_contrib, t_stats] = objective_rough_heston_RAUX(coeffs, par, invVCV, W, sim_cfg, kernel_table)
% OBJECTIVE_ROUGH_HESTON_RAUX  Indirect-inference objective for the rough
% Heston model with the AUGMENTED auxiliary (RAUX, 11 moments) instead of
% the LHAR (8 moments). Drop-in replacement for objective_rough_heston_LHAR:
% identical Monte-Carlo machinery (common random numbers, antithetics,
% parfor, feasibility guard, chi^2 decomposition); the ONLY change is that
% each simulated path is summarized by rough_aux_estimate (HAR slopes +
% log-RV variogram + semivariance/lag-1 leverage + return skew/kurt) rather
% than LHAR_estimate.
%
% INPUTS:
%   coeffs       - (11 x 1) target RAUX moments (empirical, from setup)
%   par          - (1 x 5) free parameters [lambda, theta, nu, rho, H]
%   invVCV       - (11 x 11) inverse moment covariance (block-bootstrap),
%                  used as the efficient weighting matrix
%   W            - struct .seeds (n_indep x 1), .M_fine, .ndays_total, .n_indep
%   sim_cfg      - struct .ndays_eval, .ndays_burn, .M_fine, .agg_factor, .annualize
%   kernel_table - struct from precompute_kernel_table
%
% OUTPUTS (same contract as objective_rough_heston_LHAR):
%   y            - SMM weighted quadratic loss betamean' * invVCV * betamean
%   betamean     - (11 x 1) mean (simulated - empirical) moment gap
%   stdbeta      - (11 x 1) MC standard error of the mean across replications
%   chi2_contrib - (11 x 1) per-moment chi^2 decomposition b .* (W*b)
%   t_stats      - (11 x 1) pseudo t-stat betamean ./ stdbeta

PENALTY = 1e10;

% ------------------ unpack ------------------
p.lambda = par(1);
p.theta  = par(2);
p.nu     = par(3);
p.rho    = par(4);
p.H      = par(5);
p.mu     = 0;

n_indep      = W.n_indep;
n_repl_total = 2 * n_indep;
nmom         = length(coeffs);
M_fine_loc   = W.M_fine;
T_total_loc  = W.ndays_total;
seeds_loc    = W.seeds;

% ------------------ feasibility ------------------
% Same thresholds as objective_rough_heston_LHAR (2026-05-21 relaxation):
% Feller >= 1e-4 keeps the variance well-defined; nu/lambda <= 500 rules out
% the pathological random-walk regime without compressing lambda.
feller = 2 * p.theta / (p.nu^2);
if feller < 1e-4 || p.nu / p.lambda > 500
    y = PENALTY;
    betamean     = NaN(nmom, 1);
    stdbeta      = NaN(nmom, 1);
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

% ------------------ inner Monte Carlo loop ------------------
beta = NaN(nmom, n_repl_total);

parfor j = 1:n_repl_total
    beta_j = NaN(nmom, 1);
    try
        if j <= n_indep
            j_idx = j;
            s_val = +1;
        else
            j_idx = j - n_indep;
            s_val = -1;
        end

        % Common random numbers + antithetics: same seed -> same path across
        % optimizer iterations; s_val flips the sign for the antithetic wave.
        rs = RandStream('Threefry', 'Seed', seeds_loc(j_idx));
        w_var_j  = s_val * randn(rs, M_fine_loc, T_total_loc);
        w_perp_j = s_val * randn(rs, M_fine_loc, T_total_loc);

        r5min = simulate_rough_heston(...
            sim_cfg.ndays_eval, sim_cfg.ndays_burn, ...
            sim_cfg.M_fine, sim_cfg.agg_factor, ...
            p, kernel_table, w_var_j, w_perp_j);

        RV       = sum(r5min.^2, 1) * sim_cfg.annualize;
        dailyret = sum(r5min, 1)    * sqrt(sim_cfg.annualize);

        if all(isfinite(RV)) && all(RV > 0) && all(isfinite(dailyret))
            logRV = log(RV);
            if all(isfinite(logRV))
                % r5min is (intervals x days); rough_aux_estimate wants
                % (days x intervals), so pass the transpose. logRV/dailyret
                % are rows -> pass as columns.
                result = rough_aux_estimate(logRV', dailyret', r5min');
                if all(isfinite(result.moments))
                    beta_j = result.moments - coeffs;
                end
            end
        end
    catch
        % beta_j stays NaN
    end
    beta(:, j) = beta_j;
end

n_valid = sum(all(isfinite(beta), 1));

if n_valid < max(2, 0.25 * n_repl_total)
    y = PENALTY;
    betamean     = NaN(nmom, 1);
    stdbeta      = NaN(nmom, 1);
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

valid_cols = all(isfinite(beta), 1);
betamean = mean(beta(:, valid_cols), 2);
stdbeta  = std(beta(:, valid_cols), 0, 2) / sqrt(sum(valid_cols));

if any(~isfinite(betamean))
    y = PENALTY;
    chi2_contrib = NaN(nmom, 1);
    t_stats      = NaN(nmom, 1);
    return
end

y = betamean' * invVCV * betamean;
if ~isfinite(y)
    y = PENALTY;
end

% Per-moment chi^2 decomposition: y = sum_i b(i) * (W*b)(i). With W
% non-diagonal individual entries can be negative; their magnitude still
% flags which moment is being sacrificed.
Wb           = invVCV * betamean;
chi2_contrib = betamean .* Wb;
t_stats      = betamean ./ max(stdbeta, 1e-12);

end
