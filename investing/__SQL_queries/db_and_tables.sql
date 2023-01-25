CREATE TABLESPACE safehaven OWNER postgres LOCATION 'E:\PycharmProjects\safehaven_investing\data\dbs';

CREATE DATABASE database_name
WITH
   [OWNER =  role_name]
   [TEMPLATE = template]
   [ENCODING = encoding]
   [LC_COLLATE = collate]
   [LC_CTYPE = ctype]
   [TABLESPACE = tablespace_name]
   [ALLOW_CONNECTIONS = true | false]
   [CONNECTION LIMIT = max_concurrent_connection]
   [IS_TEMPLATE = true | false ]


CREATE DATABASE safehaven_investing
WITH OWNER = postgres ENCODING = 'UTF8' TABLESPACE = safehaven;

DROP TABLE IF EXISTS yahoo_ohlc_daily_TEST;

CREATE TABLE IF NOT EXISTS yahoo_ohlc_daily_TEST (
  yahoo_ohlc_daily_id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  datetime DATE NOT NULL,
  symbol VARCHAR(45) NULL,
  open FLOAT,
  high FLOAT,
  low FLOAT,
  close FLOAT,
  adj_close FLOAT,
  volume INTEGER,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ)
TABLESPACE safehaven;


/* Hauptteil: über psycopg2 und sqlalchemy mit der DB reden */

-- testen mit simple.py
-- hidden.py anlegen
-- gitignore pflegen