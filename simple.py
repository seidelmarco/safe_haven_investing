# install psycopg2 (if needed)
# pip3 install psycopg2    # (or pip)

# https://www.pg4e.com/code/simple.py

# https://www.pg4e.com/code/hidden-dist.py
# copy hidden.py to hidden.py
# edit hidden.py and put in your credentials

# python3 simple.py

# To check the results, use psql and look at the
# pythonfun table

import psycopg2
import hidden

# Load the secrets
secrets = hidden.secrets_safehaven()

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

conn.commit()   # Flush it all to the DB server

sql = 'CREATE TABLE pythonfun (id SERIAL, line TEXT);'
print(sql)
cur.execute(sql)

conn.commit()    # Flush it all to the DB server

for i in range(10):
    txt = "Have a nice day "+str(i+1)
    sql = 'INSERT INTO pythonfun (line) VALUES (%s);'
    cur.execute(sql, (txt, ))

conn.commit()

sql = "SELECT id, line FROM pythonfun WHERE id=5;"
print(sql)
cur.execute(sql)

row = cur.fetchone()
if row is None :
    print('Row not found')
else:
    print('Found', row)

sql = 'INSERT INTO pythonfun (line) VALUES (%s) RETURNING id;'
cur.execute(sql, (txt, ))
id = cur.fetchone()[0]
print('New id', id)


#sql = "EXPLAIN SELECT id, doc FROM docs03 WHERE '{syntactically}' <@ string_to_array(lower(doc), ' ');"
#cur.execute(sql)
#row = cur.fetchall()
#print(row)

conn.commit()
cur.close()

