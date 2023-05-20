from datetime import datetime, date, timedelta
import psycopg2
from connect import connect
from sqlalchemy import create_engine, text, insert
from investing import hidden


# datetime object containing current date and time
def timestamp() -> str:
    """
    :return: string of day and time
    """
    now = datetime.now()
    dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
    print('Date and Time = ', dt_string)

    return dt_string


def timestamp_onlyday() -> str:
    """
    :return: string of just the day
    """
    now = datetime.now()
    d_string = now.strftime('%Y-%m-%d')
    print('Date = ', d_string)

    return d_string


def back_to_the_future() -> str:
    backone = datetime.today() - timedelta(days=1)
    backone_string = backone.strftime('%Y-%m-%d')
    print('Yesterday = ', backone)

    return backone_string


def convert_timestamp_to_onlyday(timestamp: int) -> str:
    """
    :return: string of just the day
    """
    dt_object = datetime.fromtimestamp(timestamp)
    print("dt_object =", dt_object)
    print("type(dt_object) =", type(dt_object))

    d_string = dt_object.strftime('%Y-%m-%d')
    print('Date = ', d_string)

    return d_string


def talk_to_me():
    """
    The default client is psql which you run from die psql shell
    Use: \l for listing db, \e for connection to an editor, \c for connecting to a DB e. g.

    But we better use the cozy, snug comfort of our IDE and use Python for writing a client and talking
    to our database with raw sql
    :return:
    """

    # investing and document with simple.py

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
                         'created_at TIMESTAMPTZ DEFAULT NOW());'))  # TABLESPACE safehaven
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
    echo=True for debugging, but it's disturbing in do_ml()-function
    NotImplementedError: This method is not implemented for SQLAlchemy 2.0. -> Solution: future=False
    :return:
    """
    sql_string = hidden.alchemy(hidden.secrets_safehaven())
    engine = create_engine(sql_string, echo=False, future=False)

    return engine


def tickers_list():
    tickers = ['DE', 'CMCL', 'AAPL', 'CVX', 'IMPUY', 'MTNOY', 'BLDP', 'KO', 'DLTR',
               'XOM', 'JNJ', 'KHC', 'MKC', 'MSFT', 'OGN', 'SKT', 'TDG', 'CTRA', 'TRGP', 'COP', 'FSLR', 'CAT', 'HWM',
               'MOS', 'HST', 'SLB', 'RL', 'BA', 'RL', 'TJX', 'FMC', 'UAL', 'DAL', 'NWL', 'DOW', 'COST', 'DD', 'HLT',
               'RE', 'LYB', 'HPE', 'BKNG']

    return tickers


def tickers_list_europe():
    """
    currently just the Euro Stoxx 50
    :return:
    """
    tickers = ['FLTR.IR', 'SAF.PA', 'MBG.DE', 'AD.AS', 'ISP.MI', 'IDEXF', 'SAN.PA', 'IBE.MC', 'BAS.DE', 'MUV2.DE',
               'NEL.OL', 'SIE.DE']

    return tickers


def tickers_list_africa():
    """
    currently the top250 african companies by marketcap
    :return:
    """
    tickers = ['NPSNY', 'ANGPY', 'FANDY', 'MTNOY', 'SGBLY', 'VDMCY', 'CKHGY', 'SSL', 'KIROY', 'GFI', 'IMPUY',
               'SBSW', 'AGRPY', 'SLLDY', 'AU', 'SRGHY', 'NDBKY', 'BDDDY', 'AAFRF', 'APNHY', 'RMGOF', 'EXXAF',
               'BDVSY', 'CLCGY', 'CIBEY', 'BTG', 'AFBOF', 'WLWHY', 'MCHOY', 'MRPLY', 'FHNGY', 'HMY', 'DSTZF', 'SRXXF',
               'SPPJY', 'TBLMY', 'PKPYY', 'MCB', 'TNGRF', 'BRRAY', 'NWKHY', 'TLKGY', 'PSGR',

    'EGS39061C014.CA',
    'KST.JO',
    'CML.JO',
    'TUWOY',
    'EGS745L1C014.CA',
    'KRO.JO',
    'OMN.JO',
    'TSG.JO',
    'EGS42111C012.CA',
    'KAP.JO',
    'AFE.JO',

    'DRD',

    'RCL.JO',
    'MT',
    'SGHC', 'EGS673Y1C015.CA', 'AFT.JO', 'HCI.JO', 'ADH.JO', 'JSEJF', 'AIP.JO', 'DTTLY', 'DTC.JO', 'EGS3D041C017.CA',
    '7YZ.F', 'RNRTY', 'RY1B.F', 'GRIN', 'GND.JO', 'GSH.JO', 'JSE.JO', 'THA.JO', 'RZT.F',
    'LBR.JO', 'EGS380S1C017.CA', 'DYLLF', 'EGS305I1C011.CA', 'SHG.JO'
               ]

    # raus, weil nicht von fmp kostenlos gelesen: 'IAM.PA', 'INL.JO', 'NPH.JO', 'PPH.JO', 'OMU.JO', 'GRT.JO', 'RBP.JO'
    # 'NY1.JO', 'AIL.JO', 'TCP.JO', 'LHC.JO', 'SNT.JO', 'DCP.JO', 'MTM.JO', 'EGS38191C010.CA', 'TRU.JO',
    # 'EGS48031C016.CA', 'GTCO.IL', 'ITE.JO', 'EGS37091C013.CA', 'EKHOLDING.KW', 'MTH.JO', 'EGS60171C013 - EGP.CA'
    # , 'EGS70011C019 - EGP.CA'

    return tickers


def flatten(l):
    """
    Shortcut/list comprehension for converting lists within a list to a single list containing strings

    flat_list = [item for sublist in l for item in sublist]

    Which means:
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    :param l:
    :return:
    """
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


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


def sum(arg):
    total = 0
    for val in arg:
        total += val
    return total


def push_df_to_db_replace(df, tablename: str):
    """
    You can use talk_to_me() or connect()
    :tablename: str
    :return:
    """

    # talk_to_me()

    connect()

    engine = sqlengine()

    # Todo: how to inherit if_exists to push_df_to_db-function?
    df.to_sql(tablename, con=engine, if_exists='replace', chunksize=100)


def get_nth_key(dictionary, n=0):
    """
    Case: dicts are unsorted per definition. If I want to put arbitrary ticker-symbols into a dict and want to fetch
    the values later without knowing which tickers the user put into the function, then I need to index by numbers since
    I need to know which on is event A and which on is event B (like in calculation e.g. Bayes' Law etc.
    :return:
    """
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError('dictionary index is out of range')


def get_nth_value(dictionary, n=0):
    """
    Case: dicts are unsorted per definition. If I want to put arbitrary ticker-symbols into a dict and want to fetch
    the values later without knowing which tickers the user put into the function, then I need to index by numbers since
    I need to know which on is event A and which on is event B (like in calculation e.g. Bayes' Law etc.
    :param dictionary:
    :param n:
    :return:
    """
    if n < 0:
        n += len(dictionary)

    for i, value in enumerate(dictionary.values()):
        if i == n:
            return value
    raise IndexError('dictionary index is out of range')


if __name__ == '__main__':
    argument = 3
    print(number_to_strings(argument))
    print('''Ist das die Lösung all meiner Probleme? Ich wollte meine util-Funktionen immer testen und
        musste den call dann auskommentieren. Dieser conditional block lässt das Programm nur im script laufen,
        wenn ich es als Modul importiere, dann wird dieser nested Codeblock nicht ausgeführt, 
        sondern die Funktion returned nur :-)\n
        https://realpython.com/if-name-main-python/#when-should-you-use-the-name-main-idiom-in-python
        ''')
    argument = 2
    print(f'Ziffer {argument} als geschriebenes Wort {number_to_string(argument)}')
    timestamp_onlyday()
    back_to_the_future()
    myswitch = PythonSwitch()

    print(myswitch.day(1))

    print(myswitch.day(4))

    print(f'Länge tickers Liste: {len(tickers_list())}')

    convert_timestamp_to_onlyday(1677392802)

    probabilities = {
        'p_DE': 0.57,
        'p_SPY': 0.45
    }

    for k, v in probabilities.items():
        print(k, v)
        print(type(k), type(v))

    print(get_nth_key(probabilities, 1))
    print(get_nth_value(probabilities, 1))



