import numpy as np
import sqlite3
import pickle
import time


def rotations24(polycube):
    """List all 24 rotations of the given 3d array
    https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array
    """
    def rotations4(polycube, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        for i in range(4):
            yield np.rot90(polycube, i, axes)

    yield from rotations4(polycube, (1, 2))
    yield from rotations4(np.rot90(polycube, 2, axes=(0, 2)), (1, 2))
    yield from rotations4(np.rot90(polycube, axes=(0, 2)), (0, 1))
    yield from rotations4(np.rot90(polycube, -1, axes=(0, 2)), (0, 1))
    yield from rotations4(np.rot90(polycube, axes=(0, 1)), (0, 2))
    yield from rotations4(np.rot90(polycube, -1, axes=(0, 1)), (0, 2))


def get_neighbors(i, j, k):
    """Return the neighboring positions for a given position (i, j, k)."""
    return [(i + 1, j, k), (i - 1, j, k), (i, j + 1, k), (i, j - 1, k), (i, j, k + 1), (i, j, k - 1)]


def expand(array, dir):
    if dir == 'right':
        padding_array = np.zeros((array.shape[0], array.shape[1], 1))
        new_array = np.concatenate((array, padding_array), axis=2)
    elif dir == 'left':
        padding_array = np.zeros((array.shape[0], array.shape[1], 1))
        new_array = np.concatenate((padding_array, array), axis=2)
    elif dir == 'down':
        padding_array = np.zeros((array.shape[0], 1, array.shape[2]))
        new_array = np.concatenate((array, padding_array), axis=1)
    elif dir == 'up':
        padding_array = np.zeros((array.shape[0], 1, array.shape[2]))
        new_array = np.concatenate((padding_array, array), axis=1)
    elif dir == 'away':
        padding_array = np.zeros((1, array.shape[1], array.shape[2]))
        new_array = np.concatenate((array, padding_array), axis=0)
    elif dir == 'towards':
        padding_array = np.zeros((1, array.shape[1], array.shape[2]))
        new_array = np.concatenate((padding_array, array), axis=0)
    else:
        raise ValueError('direction not valid')
    return new_array


def enumerate_cube(polycube_array):
    expanded_cubes = []
    shape = polycube_array.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if polycube_array[i, j, k] == 1:
                    for x, y, z in get_neighbors(i, j, k):
                        new_polycube = polycube_array.copy()

                        # Check out-of-bounds & expand array
                        if x < 0:
                            new_polycube = expand(polycube_array, 'towards')
                            x += 1
                        elif x > shape[0] - 1:
                            new_polycube = expand(polycube_array, 'away')

                        elif y < 0:
                            new_polycube = expand(polycube_array, 'up')
                            y += 1
                        elif y > shape[1] - 1:
                            new_polycube = expand(polycube_array, 'down')

                        elif z < 0:
                            new_polycube = expand(polycube_array, 'left')
                            z += 1
                        elif z > shape[2] - 1:
                            new_polycube = expand(polycube_array, 'right')

                        # Check if the cube position is valid and unfilled
                        elif 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2] and polycube_array[x, y, z] == 0:
                            pass  # Everything is okay, we can set the cube in the next step

                        else:
                            continue  # Move to the next iteration, as the cube is already filled

                        new_polycube[x, y, z] = 1
                        expanded_cubes.append(new_polycube)

    return expanded_cubes


def compute_enumerations(prev_cubes):
    new_polycubes = []
    cube_di = {}
    for prev_cube in prev_cubes:
        enumerated_cubes = enumerate_cube(prev_cube)
        for cube in enumerated_cubes:
            shape_bytes = bytes(str(cube.shape), 'utf-8')
            data_bytes = cube.tobytes()
            hashed_cube = hash(shape_bytes + data_bytes)
            if hashed_cube not in cube_di.keys():
                found = False
                rotated_cube_gen = rotations24(cube)
                while not found:
                    try:
                        rotated_cube = next(rotated_cube_gen)
                        rotated_shape_bytes = bytes(str(rotated_cube.shape), 'utf-8')
                        rotated_data_bytes = rotated_cube.tobytes()
                        rotated_hashed_cube = hash(rotated_shape_bytes + rotated_data_bytes)
                        if rotated_hashed_cube in cube_di.keys():
                            found = True
                    except StopIteration:
                        break
                if found is False:
                    cube_di[hashed_cube] = cube
                    new_polycubes.append(cube)
    return new_polycubes


def setup_database():
    conn = sqlite3.connect('polycubes.db')
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS polycubes (
            n INTEGER PRIMARY KEY,
            cubes BLOB
)
    ''')

    conn.commit()
    conn.close()


def save_polycubes(n, polycubes):
    conn = sqlite3.connect('polycubes.db')
    cursor = conn.cursor()
    start_time = time.time()

    binary = pickle.dumps(polycubes)

    cursor.execute("INSERT OR REPLACE INTO polycubes (n, cubes) VALUES (?, ?)", (n, binary))

    conn.commit()
    conn.close()

    end_time = time.time()
    print(f"Time taken to save polycubes for n={n}: {end_time - start_time:.2f} seconds")


def fetch_polycubes(n):
    conn = sqlite3.connect('polycubes.db')
    cursor = conn.cursor()

    cursor.execute("SELECT cubes FROM polycubes WHERE n=?", (n,))
    result = cursor.fetchone()

    # If no result is found
    if not result:
        return []

    # Convert binary data back to original polycubes form
    polycubes = pickle.loads(result[0])

    conn.close()
    return polycubes


def get_max_n():
    conn = sqlite3.connect('polycubes.db')
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(n) FROM polycubes")
    max_n = cursor.fetchone()[0]

    conn.close()
    return max_n or 0  # return 0 if no records found


def main(n):
    setup_database()
    max_n = get_max_n()

    # If the database is empty, initialize with n=1 polycube
    if not max_n:
        polycube = np.zeros((1, 1, 1))
        polycube[0, 0, 0] = 1
        save_polycubes(1, [polycube])
        max_n = 1

    # If we have already computed for the given n, just fetch it
    if n <= max_n:
        return fetch_polycubes(n)

    # If n > max_n, compute up to n polycube
    prev_polycubes = fetch_polycubes(max_n) if max_n else []
    i = max_n + 1
    while i <= n:
        print(f'computing enumerations for n={i}...')
        new_polycubes = compute_enumerations(prev_polycubes)
        # Save polycubes to the database
        save_polycubes(i, new_polycubes)

        print(f'{len(new_polycubes)} cubes')
        prev_polycubes = new_polycubes
        i += 1

    return fetch_polycubes(n)


if __name__ == "__main__":
    main(n=9)
