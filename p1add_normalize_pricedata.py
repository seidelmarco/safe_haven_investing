def normalize_pricedata(df_1col) -> float:
    '''
    df on input should contain only one column with the price data (plus dataframe index)
    :param df_1col:
    :return: y will be a new column in a dataframe - we will call it 'norm'
    '''
    min = df_1col.min()
    max = df_1col.max()
    x = df_1col
    # time series normalization part
    y = (x - min) / (max - min)
    return y
