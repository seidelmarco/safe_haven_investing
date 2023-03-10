from configparser import ConfigParser


def config(filename='database.ini', section='postgresql'):
    """

    :param filename: default is 'database.ini', you can also call other files
    :param section: default is 'postgresql', you can describe more sections (secrets, params) in your database.ini
    :return:
    """
    # create a parser
    parser = ConfigParser()
    # read config.py file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


if __name__ == '__main__':
    print(config(section='pg4e'))

