-- Pooled distribution summaries (all arms together). Quantiles use linear interpolation
-- (quantile_cont), the same definition as numpy's default.
SELECT
    'history' AS variable,
    'all customers' AS population,
    count(*) AS n,
    avg(history) AS mean,
    stddev_samp(history) AS sd,
    min(history) AS min,
    quantile_cont(history, 0.5) AS p50,
    quantile_cont(history, 0.9) AS p90,
    quantile_cont(history, 0.99) AS p99,
    quantile_cont(history, 0.999) AS p99_9,
    max(history) AS max
FROM customers
UNION ALL
SELECT
    'spend',
    'converters',
    count(*),
    avg(spend),
    stddev_samp(spend),
    min(spend),
    quantile_cont(spend, 0.5),
    quantile_cont(spend, 0.9),
    quantile_cont(spend, 0.99),
    quantile_cont(spend, 0.999),
    max(spend)
FROM customers
WHERE conversion = 1
UNION ALL
SELECT
    'recency',
    'all customers',
    count(*),
    avg(recency),
    stddev_samp(recency),
    min(recency),
    quantile_cont(recency, 0.5),
    quantile_cont(recency, 0.9),
    quantile_cont(recency, 0.99),
    quantile_cont(recency, 0.999),
    max(recency)
FROM customers;
