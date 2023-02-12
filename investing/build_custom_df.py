from p1add_pricedata_to_database import pull_df_from_db
from connect import connect
from myutils import sqlengine_pull_from_db
import pandas as pd

pd.set_option("display.max.columns", None)


def pull_df_wo_dates_from_db(sql: str):
    """

    :return: Dataframe object
    """
    connect()
    engine = sqlengine_pull_from_db()

    # an extra integer-index-column is added
    # df = pd.read_sql(sql, con=engine)
    # Column 'Date' is used as index
    df = pd.read_sql(sql, con=engine, index_col='symbol')

    return df


if __name__ == '__main__':
    df_sp500adjclose = pull_df_from_db()
    print(df_sp500adjclose)

    df_fmp_quote = pull_df_wo_dates_from_db(sql='fmp_quote')
    print(df_fmp_quote)

