from copy import deepcopy
import numpy as np
import utils


def translate_atoms(atoms, target_atom, coord=(0, 0, 0), change_own=False, trans_vector=None):
    cpy_atom = deepcopy(atoms) if not change_own else atoms
    coord = np.dot(np.reshape((1, 3), trans_vector), cpy_atom.cell).flatten() if trans_vector is not None else coord
    for idx in np.argwhere(np.array(cpy_atom.get_chemical_symbols()) == target_atom).flatten():
        cpy_atom.positions[idx] += coord
    return cpy_atom


def put_biggest_vacuum_on_specific_site(atoms, slab_atom, upper_atom, site, max_bond_length, move_slab=False,
                                        slab_start_pos=(0.5,0.5,0.5), ignore_axis='z', change_own=True):
    upper_atoms = utils.get_specific_atoms_from_atoms(atoms, upper_atom)
    middle_vacuum_pos = utils.get_middle_of_max_vacuum(upper_atoms.positions, max_bond_length, ignore_axis)
    slab_site = utils.get_idx_of_sites(site, atoms, slab_atom, slab_start_pos)
    slab_site[2] = middle_vacuum_pos[2] = 0
    if not move_slab:
        return translate_atoms(atoms, upper_atom, slab_site - middle_vacuum_pos, change_own)
    else:
        return translate_atoms(atoms, slab_atom, middle_vacuum_pos-slab_site, change_own)

