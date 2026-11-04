-- Per (segment variable, level, arm) sufficient statistics of the outcomes, in long format.
-- Segment variables are the pre-treatment ones of analysis plan §10; the recency band is
-- derived here (1-3, 4-6, 7-9, 10-12 months) and binary flags are reported as '0' / '1'.
-- Python turns the sums into the per-level tests (abkit.segments).
WITH long AS (
    SELECT arm, 'newbie' AS variable, newbie::VARCHAR AS level, visit, conversion, spend
    FROM customers
    UNION ALL SELECT arm, 'channel', channel, visit, conversion, spend FROM customers
    UNION ALL SELECT arm, 'zip_code', zip_code, visit, conversion, spend FROM customers
    UNION ALL SELECT arm, 'mens', mens::VARCHAR, visit, conversion, spend FROM customers
    UNION ALL SELECT arm, 'womens', womens::VARCHAR, visit, conversion, spend FROM customers
    UNION ALL SELECT
        arm,
        'recency',
        CASE
            WHEN recency <= 3 THEN '1_3'
            WHEN recency <= 6 THEN '4_6'
            WHEN recency <= 9 THEN '7_9'
            ELSE '10_12'
        END,
        visit, conversion, spend
    FROM customers
    UNION ALL SELECT arm, 'history_segment', history_segment, visit, conversion, spend
    FROM customers
)
SELECT
    variable,
    level,
    arm,
    count(*) AS n,
    sum(visit::DOUBLE) AS visit_sum,
    sum(visit::DOUBLE * visit::DOUBLE) AS visit_sum_sq,
    sum(conversion::DOUBLE) AS conversion_sum,
    sum(conversion::DOUBLE * conversion::DOUBLE) AS conversion_sum_sq,
    sum(spend::DOUBLE) AS spend_sum,
    sum(spend::DOUBLE * spend::DOUBLE) AS spend_sum_sq
FROM long
GROUP BY ALL
ORDER BY ALL;
