from datetime import datetime, date, timedelta
import psycopg2
from sqlalchemy import create_engine, text, insert
import hidden


# datetime object containing current date and time
def timestamp():
    now = datetime.now()
    dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
    print('Date and Time = ', dt_string)

    return dt_string


def timestamp_onlyday():
    now = datetime.now()
    d_string = now.strftime('%Y-%m-%d')
    print('Date = ', d_string)

    return d_string


def back_to_the_future():
    backone = datetime.today() - timedelta(days=1)
    backone_string = backone.strftime('%Y-%m-%d')
    print('Yesterday = ', backone)

    return backone_string


timestamp_onlyday()
back_to_the_future()


def talk_to_me():
    '''
    The default client is psql which you run from die psql shell
    Use: \l for listing db, \e for connection to an editor e. g.

    But we better use the cozy, snug comfort of our IDE and use Python for writing a client and talking
    to our database with raw sql
    :return:
    '''

    # test and document with simple.py

    # Load the secrets
    secrets = hidden.secrets_safehaven()

    # use the connect()-function
    conn = psycopg2.connect(
        host=secrets['host'],
        port=secrets['port'],
        database=secrets['database'],
        user=secrets['user'],
        password=secrets['pass'],
        connect_timeout=3
    )

    cur = conn.cursor()
    sql = 'DROP TABLE IF EXISTS pythonfun CASCADE;'
    print(sql)
    cur.execute(sql)

    conn.commit()  # Flush it all to the DB server

    cur.close()


# SQLalchemy - create the engine
def sqlengine():
    '''
    NotImplementedError: This method is not implemented for SQLAlchemy 2.0. -> Solution: future=False
    :return:
    '''
    sql_string = hidden.alchemy(hidden.secrets_safehaven())
    engine = create_engine(sql_string, echo=True, future=True)

    with engine.connect() as con:
        con.execute(text('DROP TABLE IF EXISTS hat_geklappt;'))
        con.execute(text('CREATE TABLE hat_geklappt (id INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY, text TEXT,'
                         'created_at TIMESTAMPTZ DEFAULT NOW()) TABLESPACE safehaven;'))
        con.commit()
        con.execute(
            text("INSERT INTO hat_geklappt (text) VALUES ('If you can read this, then the engine works - BRAVO!');"))
        con.commit()
        result = con.execute(text('SELECT * FROM hat_geklappt;'))
        con.commit()
        for tup in result:
            print(tup)

    return engine


def sqlengine_pull_from_db():
    '''
    NotImplementedError: This method is not implemented for SQLAlchemy 2.0. -> Solution: future=False
    :return:
    '''
    sql_string = hidden.alchemy(hidden.secrets_safehaven())
    engine = create_engine(sql_string, echo=True, future=False)

    return engine
