-- View: view_nasdaq_returns
-- Calculates logarithmic returns: r_t = ln(P_t / P_{t-1})
-- Handles division by zero gracefully.

DROP VIEW IF EXISTS view_nasdaq_returns CASCADE;

CREATE VIEW view_nasdaq_returns AS
SELECT
    date,
    ticker,
    adj_close,
    LN(
        adj_close / NULLIF(LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date), 0)
    ) AS log_return
FROM
    nasdaq_prices
WHERE
    adj_close > 0;