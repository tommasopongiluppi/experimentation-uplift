-- Number of customers per arm (input to the sample ratio mismatch test).
SELECT arm, count(*) AS n
FROM customers
GROUP BY arm
ORDER BY arm;
