-- Per-arm sufficient statistics of every pre-treatment variable, in long format.
-- Numeric variables give one row each (level = NULL); categorical variables give one 0/1
-- indicator row per level. Outcomes are deliberately absent.
WITH numeric_long AS (
    SELECT arm, 'recency' AS variable, NULL::VARCHAR AS level, recency::DOUBLE AS value FROM customers
    UNION ALL SELECT arm, 'history', NULL, history FROM customers
    UNION ALL SELECT arm, 'mens', NULL, mens::DOUBLE FROM customers
    UNION ALL SELECT arm, 'womens', NULL, womens::DOUBLE FROM customers
    UNION ALL SELECT arm, 'newbie', NULL, newbie::DOUBLE FROM customers
),
categorical AS (
    SELECT arm, 'history_segment' AS variable, history_segment AS level FROM customers
    UNION ALL SELECT arm, 'zip_code', zip_code FROM customers
    UNION ALL SELECT arm, 'channel', channel FROM customers
),
levels AS (
    SELECT DISTINCT variable, level FROM categorical
),
categorical_long AS (
    -- one row per customer and level: 1 if the customer has that level, else 0
    SELECT c.arm, l.variable, l.level, (c.level = l.level)::DOUBLE AS value
    FROM categorical AS c
    JOIN levels AS l USING (variable)
),
long AS (
    SELECT * FROM numeric_long
    UNION ALL
    SELECT * FROM categorical_long
)
SELECT
    variable,
    level,
    arm,
    count(*) AS n,
    sum(value) AS total,
    avg(value) AS mean,
    var_samp(value) AS variance
FROM long
GROUP BY variable, level, arm
ORDER BY variable, level NULLS FIRST, arm;
