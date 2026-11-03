-- Pooled count and share of customers by months since last purchase.
SELECT
    recency,
    count(*) AS n,
    count(*) / sum(count(*)) OVER () AS share
FROM customers
GROUP BY recency
ORDER BY recency;
