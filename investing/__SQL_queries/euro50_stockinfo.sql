SELECT "Symbol", "sector", "industry", "ebitdaMargins", "profitMargins", "grossMargins", "earningsGrowth", "targetMeanPrice", "trailingEps",
"forwardEps", "trailingAnnualDividendYield", TO_CHAR(TO_TIMESTAMP("exDividendDate"), 'yyyy-mm-dd') AS "exDivDate",
"trailingPE", "forwardPE",
* FROM public.euro50_stockinfo
ORDER BY "sector", "exDivDate" DESC, "trailingPE";