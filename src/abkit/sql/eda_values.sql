-- Raw values needed for pooled histograms: history for everyone, spend for converters.
SELECT history, CASE WHEN conversion = 1 THEN spend END AS converter_spend
FROM customers;
