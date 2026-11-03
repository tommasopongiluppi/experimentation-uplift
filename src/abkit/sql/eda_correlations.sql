-- Pooled Pearson correlation of the CUPED covariate (history) with each outcome.
SELECT 'spend' AS outcome, count(*) AS n, corr(history, spend) AS pearson_r FROM customers
UNION ALL
SELECT 'visit', count(*), corr(history, visit::DOUBLE) FROM customers
UNION ALL
SELECT 'conversion', count(*), corr(history, conversion::DOUBLE) FROM customers;
