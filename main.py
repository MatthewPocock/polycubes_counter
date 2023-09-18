import numpy as np
import time
import click
from cache_utils import *


def rotations24(polycube):
    """List all 24 rotations of the given 3d array
    https://stackoverflow.com/questions/33190042/how-to-calculate-all-24-rotations-of-3d-array
    """
    def rotations4(polycube, axes):
        """List the four rotations of the given 3d array in the plane spanned by the given axes."""
        yield polycube  # no need to rotate when i=0
        for i in range(1, 4):
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


def expand_3d_array(array, dir):
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


def expand_cube(polycube_array):
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
                            new_polycube = expand_3d_array(polycube_array, 'towards')
                            x += 1
                        elif x > shape[0] - 1:
                            new_polycube = expand_3d_array(polycube_array, 'away')

                        elif y < 0:
                            new_polycube = expand_3d_array(polycube_array, 'up')
                            y += 1
                        elif y > shape[1] - 1:
                            new_polycube = expand_3d_array(polycube_array, 'down')

                        elif z < 0:
                            new_polycube = expand_3d_array(polycube_array, 'left')
                            z += 1
                        elif z > shape[2] - 1:
                            new_polycube = expand_3d_array(polycube_array, 'right')

                        # Check if the cube position is valid and unfilled
                        elif 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2] and polycube_array[x, y, z] == 0:
                            pass
                        else:
                            continue

                        new_polycube[x, y, z] = 1
                        expanded_cubes.append(new_polycube)

    return expanded_cubes


def compute_next_cubes(prev_cubes):
    new_polycubes = []
    cube_hashes = set()

    for prev_cube in prev_cubes:
        enumerated_cubes = expand_cube(prev_cube)

        for cube in enumerated_cubes:
            if not any(hash_cube(rot_cube) in cube_hashes for rot_cube in rotations24(cube)):
                new_polycubes.append(cube)
                cube_hashes.add(hash_cube(cube))

    return new_polycubes


def hash_cube(cube):
    shape_bytes = bytes(str(cube.shape), 'utf-8')
    data_bytes = cube.tobytes()
    return hash(shape_bytes + data_bytes)


@click.command()
@click.argument("n", type=int)
@click.option('--no-cache', is_flag=True, default=False, help='Do not use cache and do not cache results')
def generate_polycubes(n, no_cache):
    # Starting polycube
    polycubes = [np.array([[[1]]], dtype=np.int8)]

    # Only set up the directory and check for max_n if caching is not disabled
    if not no_cache:
        setup_directory()
        max_n = get_max_n()

        # If the directory is empty, initialize with n=1 polycube
        if not max_n:
            save_polycubes(1, polycubes)
            max_n = 1
        # if n already computed, just fetch result and return
        elif n <= max_n:
            print(f'{len(fetch_polycubes(n))} cubes')
            return
        else:
            polycubes = fetch_polycubes(max_n)
    else:
        max_n = 1  # You begin with n=1 polycube

    # If n > max_n, compute up to n polycube
    for i in range(max_n + 1, n + 1):
        start_time = time.time()
        print(f'computing enumerations for n={i}...')
        polycubes = compute_next_cubes(polycubes)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Computation for n={i} took {elapsed_time:.2f} seconds.')

        if not no_cache:
            save_polycubes(i, polycubes)

    print(f'Found {len(polycubes)} cubes')


if __name__ == "__main__":
    generate_polycubes()
