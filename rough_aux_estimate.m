function result = rough_aux_estimate(logRV, dailyret, intraday)
% ROUGH_AUX_ESTIMATE  Augmented auxiliary model for rough Heston indirect
% inference (the "RAUX" auxiliary).
%
% MOTIVATION (see README, section "ROUGH AUXILIARY (RAUX)"):
%   The LHAR auxiliary under-identifies TWO rough-Heston parameters:
%     - H   : the LHAR has no roughness channel, so the small-scale scaling
%             of log-volatility (which is what H controls) is invisible to
%             it; H ends up confounded with the mean reversion lambda
%             (both shape the autocorrelation decay), and the H profile is
%             flat. THIS is fixed by the log-RV variogram block.
%     - rho : the LHAR leverage terms (gamma_d/w/m on lagged negative
%             returns) give a weak, near-flat binding function in rho on
%             SPY (cross-restart rho spread ~ 2). THIS is fixed by the
%             realized-semivariance leverage and the lag-1 realized
%             leverage correlation.
%
% MOMENT VECTOR (result.moments, 11 x 1, result.names gives the labels):
%    1  mean_logRV    mean of log realized variance              -> theta (level)
%    2  har_beta_d    HAR daily slope                            -> persistence
%    3  har_beta_w    HAR weekly slope                           -> lambda
%    4  har_beta_m    HAR monthly slope                          -> lambda
%    5  har_var_eps   variance of HAR residuals                  -> nu
%    6  vg_slope      log-RV variogram slope (Hurst proxy)       -> H   (primary)
%    7  vg_level      log-RV variogram intercept                 -> nu / H scale
%    8  lev_corr1     corr(r_t, logRV_{t+1})  (lag-1 leverage)   -> rho  (primary)
%    9  semivar_lev   corr((RS^- - RS^+)/RV_t, logRV_{t+1})      -> rho  (semivariance)
%   10  ret_skew      skewness of daily returns                  -> rho
%   11  ret_kurt      excess kurtosis of daily returns           -> nu
%
%   Identification map (over-identified: 11 moments for 5 parameters):
%     theta  <- mean_logRV
%     lambda <- har_beta_d, har_beta_w, har_beta_m   (persistence decay)
%     nu     <- vg_level, har_var_eps, ret_kurt
%     H      <- vg_slope   (the only clean channel)
%     rho    <- lev_corr1, semivar_lev, ret_skew
%
% INPUTS (constructed IDENTICALLY on empirical and simulated data):
%   logRV    - (T x 1) log annualized realized variance, one per day
%   dailyret - (T x 1) daily returns. Only scale/shift-invariant functionals
%              are used (skew, kurtosis, correlations), so the common
%              annualization factor need not match between empirical and
%              simulated inputs - only the CONSTRUCTION must match.
%   intraday - (T x M) intraday (e.g. 5-min) returns, oriented days x
%              intervals. Used only for the signed realized-semivariance
%              ratio (RS^- - RS^+)/RV, which is scale-free, so again no
%              annualization is required. An (intervals x days) matrix is
%              transposed automatically.
%
% NOTE ON THE VARIOGRAM AND INDIRECT INFERENCE:
%   RV is a noisy estimator of integrated variance, which biases the
%   variogram at small lags. In indirect inference this bias CANCELS,
%   because the binding function is evaluated by computing the SAME
%   statistic on the simulated path (RV built from the same intraday grid).
%   We therefore use the raw variogram as an auxiliary statistic, not as a
%   point estimator of H, and skip lag 1 (the noisiest) to keep the slope
%   sensitive to H.
%
% OUTPUT:
%   result.moments - (11 x 1) moment vector; NaN(11,1) if any block fails
%                    (the objective penalizes NaN moments).
%   result.names   - (1 x 11) cell of moment labels.

NMOM = 11;
result.names = {'mean_logRV','har_beta_d','har_beta_w','har_beta_m', ...
                'har_var_eps','vg_slope','vg_level','lev_corr1', ...
                'semivar_lev','ret_skew','ret_kurt'};
result.moments = NaN(NMOM, 1);

% ------------------ coerce shapes ------------------
logRV    = logRV(:);
dailyret = dailyret(:);
T = numel(logRV);
if size(intraday, 1) ~= T && size(intraday, 2) == T
    intraday = intraday.';   % accept (intervals x days) and fix to (days x intervals)
end
if numel(dailyret) ~= T || size(intraday, 1) ~= T || T < 100
    return                    % shape mismatch / too short -> NaN moments
end
if any(~isfinite(logRV)) || any(~isfinite(dailyret))
    return
end

% ------------------ Block A: HAR-RV on log-RV ------------------
% Reuse the existing, tested HAR estimator (same regressors as the LHAR
% pipeline, leverage terms dropped). Returns beta = [alpha; b_d; b_w; b_m]
% and the residual variance.
har = HAR_estimate(logRV);
if any(~isfinite(har.moments))
    return
end
mean_logRV  = mean(logRV);
har_beta_d  = har.beta(2);
har_beta_w  = har.beta(3);
har_beta_m  = har.beta(4);
har_var_eps = har.var_resid;

% ------------------ Block B: log-RV variogram (roughness -> H) ------------------
% m1(tau) = E|logRV_{t+tau} - logRV_t|. For a rough log-volatility the
% structure function scales as m1(tau) ~ tau^H over the relevant range, so
% the slope of log m1 against log tau estimates H. Lag 1 is excluded (RV
% measurement noise dominates it and flattens the H-sensitivity).
lags = [2 5 10 22];
m1 = zeros(numel(lags), 1);
for k = 1:numel(lags)
    tau = lags(k);
    if T <= tau + 1
        return
    end
    d = abs(logRV(1+tau:end) - logRV(1:end-tau));
    m1(k) = mean(d);
end
if any(m1 <= 0) || any(~isfinite(m1))
    return
end
pcoef    = polyfit(log(lags(:)), log(m1), 1);
vg_slope = pcoef(1);
vg_level = pcoef(2);

% ------------------ Block C: lag-1 realized leverage (-> rho) ------------------
% corr(r_t, logRV_{t+1}): for equity leverage (rho<0) a low return today
% predicts higher variance tomorrow, so this correlation is negative.
lev_corr1 = local_corr(dailyret(1:end-1), logRV(2:end));

% ------------------ Block C2: signed realized semivariance leverage (-> rho) ------------------
% RS^+ / RS^- are the up/down realized semivariances (Barndorff-Nielsen et
% al.; Patton-Sheppard). The signed ratio (RS^- - RS^+)/RV is in (-1,1) and
% scale-free. Correlating it with next-day log-RV captures the asymmetric
% ("bad volatility predicts volatility") leverage channel, a much stronger
% identifier of rho than the LHAR gamma_* terms.
rv_raw  = sum(intraday.^2, 2);
rsp_raw = sum((intraday .* (intraday > 0)).^2, 2);
rsm_raw = sum((intraday .* (intraday < 0)).^2, 2);
sv_ratio = NaN(T, 1);
pos = rv_raw > 0;
sv_ratio(pos) = (rsm_raw(pos) - rsp_raw(pos)) ./ rv_raw(pos);
semivar_lev = local_corr(sv_ratio(1:end-1), logRV(2:end));

% ------------------ Block D: return distribution shape ------------------
% Negative skew -> rho<0; excess kurtosis grows with the vol-of-vol nu.
m  = mean(dailyret);
sd = std(dailyret, 1);          % population std (normalization by N)
if sd <= 0 || ~isfinite(sd)
    return
end
z = (dailyret - m) / sd;
ret_skew = mean(z.^3);
ret_kurt = mean(z.^4) - 3;      % excess kurtosis

% ------------------ assemble ------------------
mom = [mean_logRV; har_beta_d; har_beta_w; har_beta_m; har_var_eps; ...
       vg_slope;    vg_level;   lev_corr1;  semivar_lev; ...
       ret_skew;    ret_kurt];
if all(isfinite(mom))
    result.moments = mom;
end

end


% ===================== local helper =====================
function c = local_corr(a, b)
% Pearson correlation, toolbox-free, NaN-robust. Returns NaN if fewer than
% 3 paired finite observations or if either series is constant.
    a = a(:); b = b(:);
    good = isfinite(a) & isfinite(b);
    a = a(good); b = b(good);
    if numel(a) < 3
        c = NaN; return
    end
    sa = std(a, 1); sb = std(b, 1);
    if sa <= 0 || sb <= 0 || ~isfinite(sa) || ~isfinite(sb)
        c = NaN; return
    end
    c = mean((a - mean(a)) .* (b - mean(b))) / (sa * sb);
end
