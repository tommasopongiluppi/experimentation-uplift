-- Row-level pre-treatment covariates and outcomes: input to the Lin (2013) regression
-- adjustment, which needs one row per customer. CUPED itself runs on the per-arm sums in
-- arm_metrics.sql. spend_winsorised uses the same pooled cap as arm_metrics.sql: the 95th
-- percentile of converters' spend across all arms (analysis plan §8).
WITH cap AS (
    SELECT quantile_cont(spend, 0.95) AS spend_cap
    FROM customers
    WHERE conversion = 1
)
SELECT
    c.customer_id,
    c.arm,
    c.recency,
    c.history_segment,
    c.history,
    c.mens,
    c.womens,
    c.zip_code,
    c.newbie,
    c.channel,
    c.visit::DOUBLE AS visit,
    c.conversion::DOUBLE AS conversion,
    c.spend::DOUBLE AS spend,
    least(c.spend, cap.spend_cap)::DOUBLE AS spend_winsorised
FROM customers AS c
CROSS JOIN cap
ORDER BY c.customer_id;
