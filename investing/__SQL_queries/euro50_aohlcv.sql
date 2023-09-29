SELECT "BAS.DE_Adj_Close", "MUV2.DE_Adj_Close",
* FROM public.ohlc_europe
ORDER BY "Date" DESC;

--DELETE FROM public.ohlc_europe WHERE "ADS.DE_Volume" = ;