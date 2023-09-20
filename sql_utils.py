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
        CREATE TABLE polycubes (
            id INTEGER PRIMARY KEY,
            n INTEGER NOT NULL,
            data BLOB NOT NULL
        );
    ''')
    con.commit()

    # TODO: create table to track complete n & incomplete values of n

    return con


def save_polycubes(n, polycubes):
    with open(f'polycube_cache/polycube_{n}.pkl', 'wb') as f:
        pickle.dump(polycubes, f)

    print(f"Saved polycubes for n={n}")


def fetch_polycubes(n):
    con = sqlite3.connect("polycubes.db")
    cur = con.cursor()

    cur.execute("SELECT data FROM polycubes WHERE n=?", (n,))
    blobs = cur.fetchall()
    polycubes = tuple(bytes_to_numpy(blob[0]) for blob in blobs)

    con.close()
    return polycubes


def get_max_n():
    saved_files = [f for f in os.listdir('polycube_cache') if f.startswith('polycube_') and f.endswith('.pkl')]
    if not saved_files:
        return 0
    # Extract the numbers from filenames and get the max
    max_n = max([int(file.split('_')[1].split('.')[0]) for file in saved_files])
    return max_n
