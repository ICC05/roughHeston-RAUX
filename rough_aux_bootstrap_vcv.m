function [coeffs, VCV, boot_moments, n_ok] = rough_aux_bootstrap_vcv(logRV, dailyret, intraday, block_len, B, seed)
% ROUGH_AUX_BOOTSTRAP_VCV  Moving-block bootstrap covariance of the RAUX
% auxiliary moment vector, used as the efficient indirect-inference
% weighting matrix W = inv(VCV).
%
% Several RAUX moments (mean log-RV, variogram slope/intercept, the
% leverage correlations, return skewness/kurtosis) are NOT OLS coefficients,
% so their joint covariance has no closed form like the Newey-West VCV of
% the LHAR regression. The moving-block bootstrap estimates it directly
% while preserving the serial dependence of the daily series.
%
% IMPORTANT: day-blocks are resampled JOINTLY across logRV, dailyret and
% intraday (the same day indices index all three), so the cross-series
% dependence that the leverage moments rely on is preserved.
%
% INPUTS:
%   logRV     - (T x 1) empirical log realized variance
%   dailyret  - (T x 1) empirical daily returns
%   intraday  - (T x M) empirical intraday returns (days x intervals)
%   block_len - moving-block length in days (>= max auxiliary lag, ~44)
%   B         - number of bootstrap resamples (~1000)
%   seed      - RNG seed (Threefry) for reproducibility
%
% OUTPUTS:
%   coeffs       - (NMOM x 1) point estimate of the moment vector on the
%                  full empirical sample (= rough_aux_estimate(...).moments)
%   VCV          - (NMOM x NMOM) bootstrap covariance, ridge-regularized
%   boot_moments - (n_ok x NMOM) the successful bootstrap moment vectors
%   n_ok         - number of bootstrap resamples that produced finite moments

logRV    = logRV(:);
dailyret = dailyret(:);
T = numel(logRV);
if size(intraday, 1) ~= T && size(intraday, 2) == T
    intraday = intraday.';
end

% Point estimate on the full sample.
base   = rough_aux_estimate(logRV, dailyret, intraday);
coeffs = base.moments;
nmom   = numel(coeffs);

if any(~isfinite(coeffs))
    error('rough_aux_bootstrap_vcv:bad_base', ...
        'rough_aux_estimate returned non-finite moments on the empirical sample.');
end

% Moving-block bootstrap.
nblocks   = ceil(T / block_len);
max_start = T - block_len + 1;
if max_start < 1
    error('rough_aux_bootstrap_vcv:block_too_long', ...
        'block_len (%d) exceeds sample length T (%d).', block_len, T);
end

rs = RandStream('Threefry', 'Seed', seed);
boot_moments = NaN(B, nmom);

for b = 1:B
    starts = randi(rs, max_start, nblocks, 1);
    idx = zeros(nblocks * block_len, 1);
    for j = 1:nblocks
        idx((j-1)*block_len + (1:block_len)) = starts(j) + (0:block_len-1);
    end
    idx = idx(1:T);
    mb  = rough_aux_estimate(logRV(idx), dailyret(idx), intraday(idx, :));
    boot_moments(b, :) = mb.moments(:)';
end

good = all(isfinite(boot_moments), 2);
boot_moments = boot_moments(good, :);
n_ok = size(boot_moments, 1);

if n_ok < nmom + 1
    error('rough_aux_bootstrap_vcv:too_few', ...
        'Only %d/%d bootstrap resamples gave finite moments; cannot form VCV.', n_ok, B);
end

VCV = cov(boot_moments);

% Ridge-regularize so the inverse is well conditioned even if two leverage
% moments are nearly collinear (intended: they over-identify rho).
ridge = 1e-8 * mean(diag(VCV));
VCV   = VCV + ridge * eye(nmom);

end
