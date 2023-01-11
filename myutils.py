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


def talk_to_me():
    """
    The default client is psql which you run from die psql shell
    Use: \l for listing db, \e for connection to an editor e. g.

    But we better use the cozy, snug comfort of our IDE and use Python for writing a client and talking
    to our database with raw sql
    :return:
    """

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
    """
    NotImplementedError: This method is not implemented for SQLAlchemy 2.0. -> Solution: future=False
    :return:
    """
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
    """
    NotImplementedError: This method is not implemented for SQLAlchemy 2.0. -> Solution: future=False
    :return:
    """
    sql_string = hidden.alchemy(hidden.secrets_safehaven())
    engine = create_engine(sql_string, echo=True, future=False)

    return engine


# Function to convert number into string
# Switcher is dictionary data type here
def number_to_strings(argument):
    '''
    get() method of dictionary data type returns value of passed argument if it is present in dict otherwise
    second argument will be assigned as default value of passed argument
    :param argument: key from dict
    :return:
    '''
    switcher = {
        0: 'zero',
        1: 'one',
        2: 'two',
    }

    return switcher.get(argument, 'nothing')


if __name__ == '__main__':
    argument = 3
    print(number_to_strings(argument))
    print('''Ist das die Lösung all meiner Probleme? Ich wollte meine util-Funktionen immer testen und
    musste den call dann auskommentieren. Dieser conditional block lässt das Programm nur im script laufen,
    wenn ich es als Modul importiere, dann wird dieser nested Codeblock nicht ausgeführt, 
    sondern die Funktion returned nur :-)\n
    https://realpython.com/if-name-main-python/#when-should-you-use-the-name-main-idiom-in-python
    ''')


# Method 2 for switches:
def bike_switcher(bike):
    """

    :param bike:
    :return:
    """

    if bike == 'Hero':
        print('bike is Hero')

    elif bike == 'Suzuki':
        print('bike is Suzuki')

    elif bike == 'Yamaha':
        print('bike is Yamaha')

    else:
        print('Please choose correct answer')


if __name__ == '__main__':
    bike = 'Hero'
    print(bike_switcher(bike))

# method 3 - using a class to create a switcher


class PythonSwitch:
    def day(self, month):

        default = 'Incorrect day'

        return getattr(self, 'case_' + str(month), lambda: default)()

    @staticmethod
    def case_1():
        return 'Jan'

    @staticmethod
    def case_2():
        return 'Feb'

    @staticmethod
    def case_3():
        return 'Mar'


myswitch = PythonSwitch()

print(myswitch.day(1))

print(myswitch.day(4))


# This code runs only in Python 3.10 or above versions:
def number_to_string(argument: int):
    """
    NEW :-) Switch Case in Python
    :param argument: 
    :return: 
    """
    match argument:
        case 0:
            return 'zero'
        case 1:
            return 'one'
        case 2:
            return 'two'
        case default:
            return 'something - hihi, your number is not defined ;-)'


if __name__ == '__main__':
    argument = 2
    print(f'Ziffer {argument} als geschriebenes Wort {number_to_string(argument)}')
    timestamp_onlyday()
    back_to_the_future()


