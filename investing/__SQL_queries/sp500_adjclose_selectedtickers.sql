DO $$ 
DECLARE
   ticker    VARCHAR(5) := 'NVDA';
   created_at TIME := NOW();
BEGIN 
   RAISE NOTICE 'Der aktuelle Ticker heißt: % ', ticker;
   RAISE NOTICE 'Retrieved at: % ', created_at;
   --PERFORM pg_sleep(10);
END $$;


SELECT "Date", "AKAM", "NVDA", "RL", "CTRA", "DE", "XOM", "CVX", "AAPL", "MSFT",
"KO", "DLTR", "JNJ", "KHC", "MKC", "TDG", "BKNG", "ATVI", "AMD", "ANSS",
* FROM public.sp500_adjclose
ORDER BY "Date" DESC;



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