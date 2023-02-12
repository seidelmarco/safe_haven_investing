SELECT "sector", "industry", "shortName", * FROM public.sp500_predicted_buys AS pred
LEFT JOIN public.sp500_stockinfo AS inf ON pred."Symbol" = inf."Symbol"
ORDER BY pred."Buy_predicted" DESC;