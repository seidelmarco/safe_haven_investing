import psycopg2


def talk_to_me():
    '''
    The default client is psql which you run from die psql shell
    Use: \l for listing db, \e for connection to an editor e. g.

    But we better use the cozy, snug comfort of our IDE and use Python for writing a client and talking
    to our database with raw sql
    :return:
    '''

    # test and document with simple.py
    # from simple import ....(function)

    conn = psycopg2.connect(
        host="localhost",
        database="suppliers",
        user="postgres",
        password="Abcd1234")
