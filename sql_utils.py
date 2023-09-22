import os
import pickle
import sqlite3
import io
import numpy as np


def bytes_to_numpy(bytes_data):
    buffer = io.BytesIO(bytes_data)
    return np.load(buffer)


def setup_db():
    con = sqlite3.connect("polycubes.db")
    cur = con.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS polycubes (
            id INTEGER PRIMARY KEY,
            n INTEGER NOT NULL,
            data BLOB NOT NULL
        );
    ''')
    con.commit()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS status (
            n INTEGER PRIMARY KEY,
            completed BOOLEAN NOT NULL  
        );
    ''')

    cur.execute('''
        INSERT OR IGNORE INTO status (n, completed)
        VALUES (0, True);
    ''')

    cur.execute('''
        DELETE FROM polycubes
        WHERE n NOT IN (SELECT n FROM status WHERE completed = True);
    ''')

    con.commit()

    return con


def fetch_polycubes(n):
    con = sqlite3.connect("polycubes.db")
    cur = con.cursor()

    cur.execute("SELECT data FROM polycubes WHERE n=?", (n,))
    blobs = cur.fetchall()
    polycubes = tuple(bytes_to_numpy(blob[0]) for blob in blobs)

    con.close()
    return polycubes


def get_max_n():
    con = sqlite3.connect("polycubes.db")
    cur = con.cursor()

    cur.execute("SELECT MAX(n) FROM polycubes")
    result = cur.fetchone()
    con.close()

    if result and result[0] is not None:
        return result[0]
    else:
        return 0


def count_polycubes(n):
    con = sqlite3.connect("polycubes.db")

    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM polycubes WHERE n=?", (n,))
    count = cur.fetchone()[0]
    con.close()

    return count
