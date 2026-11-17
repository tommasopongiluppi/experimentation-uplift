-- Per-arm sufficient statistics for the outcome metrics, the CUPED covariate and spend among
-- converters. Python turns these sums into means, variances and covariances (abkit.metrics).
--
-- Parameter $winsorise (BOOLEAN): when true, spend is capped at the 95th percentile of
-- converters' spend pooled across arms (analysis plan §8). The cap is computed here, once, on
-- all converters together, never per arm. quantile_cont interpolates linearly between order
-- statistics (the numpy / pandas default). Zero spend is unchanged because the cap is >= 0.
WITH cap AS (
    SELECT quantile_cont(spend, 0.95) AS spend_cap
    FROM customers
    WHERE conversion = 1
),
base AS (
    SELECT
        c.arm,
        c.history::DOUBLE AS x,
        c.visit::DOUBLE AS visit,
        c.conversion::DOUBLE AS conversion,
        CASE WHEN $winsorise THEN least(c.spend, cap.spend_cap) ELSE c.spend END::DOUBLE AS spend,
        cap.spend_cap
    FROM customers AS c
    CROSS JOIN cap
)
SELECT
    arm,
    count(*) AS n,
    -- outcomes
    sum(visit) AS visit_sum,
    sum(visit * visit) AS visit_sum_sq,
    sum(conversion) AS conversion_sum,
    sum(conversion * conversion) AS conversion_sum_sq,
    sum(spend) AS spend_sum,
    sum(spend * spend) AS spend_sum_sq,
    -- covariate history and its cross-products with each outcome (CUPED)
    sum(x) AS history_sum,
    sum(x * x) AS history_sum_sq,
    sum(x * visit) AS history_x_visit,
    sum(x * conversion) AS history_x_conversion,
    sum(x * spend) AS history_x_spend,
    -- spend among converters
    count(*) FILTER (WHERE conversion = 1) AS n_converters,
    coalesce(sum(spend) FILTER (WHERE conversion = 1), 0) AS conv_spend_sum,
    coalesce(sum(spend * spend) FILTER (WHERE conversion = 1), 0) AS conv_spend_sum_sq,
    -- the cap applied (NULL when spend is raw)
    CASE WHEN $winsorise THEN any_value(spend_cap) END AS spend_cap
FROM base
GROUP BY arm
ORDER BY arm;
