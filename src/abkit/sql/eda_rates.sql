-- Pooled outcome rates over all customers (all arms together, never split by arm).
SELECT
    count(*) AS n_customers,
    sum(visit) AS n_visitors,
    avg(visit) AS visit_rate,
    sum(conversion) AS n_converters,
    avg(conversion) AS conversion_rate,
    avg((spend = 0)::DOUBLE) AS share_zero_spend,
    avg(spend) AS mean_spend
FROM customers;
