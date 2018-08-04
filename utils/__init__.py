import numpy as np
from copy import deepcopy
import itertools


def to_tuple(item):
    return item if isinstance(item, tuple) else tuple([item])


def get_unit_vector(vec):
    return vec / np.linalg.norm(vec)


def get_angle(v1, v2):
    return np.arccos(np.clip(np.dot(get_unit_vector(v1), get_unit_vector(v2)), -1.0, 1.0))


def get_poly_area(x_arr, y_arr):
    return 0.5 * np.abs(np.dot(x_arr, np.roll(y_arr, 1)) - np.dot(y_arr, np.roll(x_arr, 1)))


def check_ccw(p1, p2, p3):
    x_arr, y_arr = np.vstack((p1, p2, p3)).transpose()
    area = np.round(np.dot(x_arr, np.roll(y_arr, 1)) - np.dot(y_arr, np.roll(x_arr, 1)),10)
    return int(np.piecewise(area, [area < 0, area == 0, area > 0], [-1, 0, 1]))


def get_distance(p1, p2):
    return np.sqrt(sum(map(lambda x, y: (x - y) ** 2, p1, p2)))


def find_polygon(pos, idx_l, adj, bol, first_point, u, v ):
    if first_point == v:
        bol[u][v] = False
        return
    idx_l.append(v)
    bol[u][v] = False
    w = sorted([i for i in adj[v] if check_ccw(pos[u], pos[v], pos[i]) == 1], key=lambda x:
               get_angle(pos[v] - pos[u], pos[v] - pos[x]))
    if len(w) == 0:
        return
    find_polygon(pos, idx_l, adj, bol, first_point, v, w[0])


def get_middle_of_max_vacuum(positions,  max_bond_length, ignore_axis='z', search_from=None):
    if search_from is None:
        search_from = np.array([(min(positions[:, i])+max(positions[:, i]))/2 for i in range(3)])
    ignore_idx = ord(ignore_axis.lower())-120
    pos_for_cal = np.delete(positions, ignore_idx, 1)
    bol_arr = np.full((positions.shape[0], positions.shape[0]), False)
    adj = [[] for _ in range(positions.shape[0])]
    for i in range(len(adj)):
        for j in range(i+1, len(adj)):
            if get_distance(positions[i], positions[j]) <= max_bond_length:
                adj[i].append(j)
                adj[j].append(i)
                bol_arr[j][i] = bol_arr[i][j] = True
    poly_idx_arr = []
    for idx in range(len(adj)):
        for idx2 in adj[idx]:
            if bol_arr[idx][idx2]:
                first_point = idx
                idx_l = [first_point]
                find_polygon(pos_for_cal, idx_l, adj, bol_arr, first_point, idx, idx2)
                if len(idx_l) >= 3:
                    poly_idx_arr.append(idx_l)
    area_arr = np.array([get_poly_area(*tuple(np.vstack(tuple([pos_for_cal[i] for i in j])).transpose()))
                         for j in poly_idx_arr])
    max_area_idx = np.argwhere(area_arr == np.amax(area_arr)).flatten()
    real_pos = np.array([[positions[idx2] for idx2 in poly_idx_arr[idx]] for idx in max_area_idx])
    final_idx_arr = np.array([np.sum(arr, axis=0)/len(arr) for arr in real_pos])
    return sorted(final_idx_arr, key=lambda x: get_distance(x, search_from))[0]


def get_average_of_atoms(pos_idx, atoms):
    return sum([atoms.positions[i-1] for i in to_tuple(pos_idx)])/ len(to_tuple(pos_idx))


def get_specific_atoms_from_atoms(atoms, target_atom, change_own=False):
    cpy_atoms = deepcopy(atoms) if not change_own else atoms
    del cpy_atoms[[atom.index for atom in cpy_atoms if atom.symbol not in to_tuple(target_atom)]]
    return cpy_atoms


def get_idx_of_sites(site_name, atoms, target_atom, start_positions=(0.5, 0.5, 0.5)):
    pos = get_specific_atoms_from_atoms(atoms, target_atom).positions
    unique_z = np.unique(pos[:, 2])
    start_positions = np.dot(start_positions, atoms.cell).flatten()
    name_to_index = {'bridge': 2, 'fcc': 0, 'top': 2, 'hcp': 1}
    start_positions[2] = unique_z[name_to_index[site_name]]
    new_pos = np.array([i for i in pos if i[2] == unique_z[name_to_index[site_name]]])
    if site_name == 'bridge':
        min_val = min(map(lambda x: get_distance(*x), itertools.combinations(new_pos,2)))
        final_pos = [x for x in itertools.combinations(new_pos,2) if min_val < get_distance(*x) < min_val + 0.001]
        final_pos = np.array([(x[0]+x[1])/2 for x in final_pos])
        return final_pos[np.argmin(np.array(list(map(lambda x: get_distance(x, start_positions), final_pos))))]
    return new_pos[np.argmin(np.array(list(map(lambda x: get_distance(x, start_positions), new_pos))))]
