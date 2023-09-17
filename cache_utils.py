import os
import pickle


def setup_directory():
    if not os.path.exists('polycube_cache'):
        os.makedirs('polycube_cache')


def save_polycubes(n, polycubes):
    with open(f'polycube_cache/polycube_{n}.pkl', 'wb') as f:
        pickle.dump(polycubes, f)

    print(f"Saved polycubes for n={n}")


def fetch_polycubes(n):
    filepath = f'polycube_cache/polycube_{n}.pkl'

    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        return []


def get_max_n():
    saved_files = [f for f in os.listdir('polycube_cache') if f.startswith('polycube_') and f.endswith('.pkl')]
    if not saved_files:
        return 0
    # Extract the numbers from filenames and get the max
    max_n = max([int(file.split('_')[1].split('.')[0]) for file in saved_files])
    return max_n
