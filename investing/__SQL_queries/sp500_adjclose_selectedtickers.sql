DO $$ 
DECLARE
   ticker    VARCHAR(5) := 'NVDA';
   created_at TIME := NOW();
BEGIN 
   RAISE NOTICE 'Der aktuelle Ticker heißt: % ', ticker;
   RAISE NOTICE 'Retrieved at: % ', created_at;
   --PERFORM pg_sleep(10);
END $$;


SELECT "ID", "Date", "AKAM", "NVDA", "RL", "CTRA", "DE", "XOM", "CVX", "AAPL", "MSFT",
"KO", "DLTR", "JNJ", "KHC", "MKC", "TDG", "BKNG", "ATVI", "AMD", "ANSS",
* FROM public.sp500_adjclose
ORDER BY "Date" DESC;

-- wir haben in adjclose 464 Reihen, in volume aber 465 - finde die Dublette oder extra Zeile, wahrscheinlich
-- haben wir in adjclose mal einen Tag vergessen
SELECT adj."Date", COUNT(*) FROM public.sp500_adjclose AS adj
RIGHT JOIN public.sp500_volume AS vol
ON adj."Date" = vol."Date"
GROUP BY adj."Date" HAVING COUNT(*) > 1
;

SELECT * FROM public.sp500_volume AS vol WHERE NOT EXISTS 
(SELECT 1 FROM public.sp500_adjclose AS adj WHERE vol."Date" = adj."Date" )
;
--Resultat: in adj gibt es den 23.01.23 mit 6 Uhr! In Vol den 23.01.23 mit 0 Uhr - ist also nicht so schlimm
-- in adj den 18.05.23 NICHT, in vol gibt es diesen Tag :-)
-- Lösung: adjclose nochmal ziehen - ne besser nicht, weil wir einen abhängigen view haben, besser den 18.05.23 nochmal
--ziehen

--DELETE FROM public.sp500_adjclose WHERE "Date" = '2023-11-08 00:00:00';
--DELETE FROM public.sp500_adjclose WHERE "ID" = 477;


ALTER TABLE public.sp500_adjclose ADD COLUMN "ID" INTEGER GENERATED ALWAYS AS IDENTITY;




CREATE OR REPLACE VIEW top_picks_10_days AS
SELECT 	"Date",
		(CASE WHEN ROW_NUMBER()
		OVER (ORDER BY "Date" DESC) >= 10
		THEN AVG("AAPL") OVER (ORDER BY "Date" DESC
                        ROWS BETWEEN 9 preceding and current ROW) END)
		AS MovingAvg10_AAPL,
		
		(CASE WHEN ROW_NUMBER()
		OVER (ORDER BY "Date" DESC) >= 10
		THEN AVG("CTRA") OVER (ORDER BY "Date" DESC
                        ROWS BETWEEN 9 preceding and current ROW) END)
		AS MovingAvg10_CTRA,
		
		(CASE WHEN ROW_NUMBER()
		OVER (ORDER BY "Date" DESC) >= 10
		THEN AVG("NVDA") OVER (ORDER BY "Date" DESC
                        ROWS BETWEEN 9 preceding and current ROW) END)
		AS MovingAvg10_NVDA,
		
		(CASE WHEN ROW_NUMBER()
		OVER (ORDER BY "Date" DESC) >= 10
		THEN AVG("DE") OVER (ORDER BY "Date" DESC
                        ROWS BETWEEN 9 preceding and current ROW) END)
		AS MovingAvg10_DE
		
		
		FROM public.sp500_adjclose;
		

CREATE OR REPLACE VIEW top_picks_30_days AS
SELECT
	"Date",
	(CASE WHEN ROW_NUMBER()
		OVER (ORDER BY "Date" DESC) >= 30
		THEN AVG("AAPL") OVER (ORDER BY "Date" DESC
						ROWS BETWEEN 29 preceding and current ROW) END)
	AS MovingAvg30_AAPL,
			
	(CASE WHEN ROW_NUMBER()
		OVER (ORDER BY "Date" DESC) >= 30
		THEN AVG("CTRA") OVER (ORDER BY "Date" DESC
                        ROWS BETWEEN 29 preceding and current ROW) END)
	AS MovingAvg30_CTRA,
		
	(CASE WHEN ROW_NUMBER()
	OVER (ORDER BY "Date" DESC) >= 30
	THEN AVG("NVDA") OVER (ORDER BY "Date" DESC
					ROWS BETWEEN 29 preceding and current ROW) END)
	AS MovingAvg30_NVDA,
		
	(CASE WHEN ROW_NUMBER()
	OVER (ORDER BY "Date" DESC) >= 30
	THEN AVG("DE") OVER (ORDER BY "Date" DESC
					ROWS BETWEEN 29 preceding and current ROW) END)
	AS MovingAvg30_DE
		
			
	FROM public.sp500_adjclose;

		
-- next step: rank the highest returns within a window, or let cross the SMA14 the SMA200 and rank the true values
		
		
		
SELECT AVG("NVDA") OVER (ORDER BY "Date" DESC
                        ROWS BETWEEN 9 preceding and current ROW) AS MovingAvg10
from public.sp500_adjclose;

SELECT COUNT("NVDA") FROM public.sp500_adjclose;
--SELECT * FROM public.top_picks_10_days;
SELECT * FROM public.top_picks_30_days;