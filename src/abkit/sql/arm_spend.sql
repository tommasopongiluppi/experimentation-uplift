-- Row-level spend per customer with its arm: input to the percentile bootstrap only.
-- Every other frequentist statistic comes from the sufficient statistics in arm_metrics.sql.
SELECT arm, spend::DOUBLE AS spend
FROM customers
ORDER BY customer_id;
