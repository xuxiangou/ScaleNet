import numpy as np
from numpy.linalg import norm
from ase.io import read, write
from ase.data import covalent_radii as covalent
from ase.neighborlist import NeighborList
from itertools import combinations
import networkx as nx
import networkx.algorithms.isomorphism as iso
from sys import argv
from os import system, path
from chem_env_sid import process_atoms, process_site, unique_chem_envs, draw_atomic_graphs, compare_chem_envs
from pathlib import Path
import os

def make_fol(path_dir,atoms):
    os.makedirs(path_dir)
    store_path = path_dir + '/' + 'POSCAR'
    atoms.write(store_path,format='vasp')

def make_fol_duplicate(j,path_dir,atoms):
    if not path.isdir(path_dir):
        os.makedirs(path_dir)
    store_path = path_dir + '/' + 'POSCAR_{}'.format(j)
    atoms.write(store_path,format='vasp')



if __name__ == "__main__":
    def natural_cutoffs(atoms, multiplier=1.1):
        """Generate a neighbor list cutoff for every atom"""
        return [covalent[atom.number] * multiplier for atom in atoms]

    from sys import argv
    from os import system

    all_atoms = [read(poscar) for poscar in argv[1:]]
    chem_envs = []
    # print(len(all_atoms))
    unique = []
    unique_id = []
    for i,atoms in enumerate(all_atoms):
################# These can be added in as arg parsers ##################
        ads = read("OH.POSCAR")
        surface_atoms = ["Pt", "Sn", "Pd"]
        radii_multiplier = 1.1
        skin_arg = 0.25
        no_adsorb = ['Sn']
        min_ads_dist = 1.4
##########################################################################

        nl = NeighborList(natural_cutoffs(atoms, radii_multiplier), self_interaction=False,  bothways=True, skin=skin_arg) ### the generation of nl here needs to be made consistent and loaded from the chem_env file.
        nl.update(atoms)

        adsorbate_atoms = [index for index, atom in enumerate(atoms) if atom.symbol not in surface_atoms]

#        print(i,atoms,nl)
        full_graph, envs = process_atoms(atoms, nl=nl, adsorbate_atoms=adsorbate_atoms,radius=3)
        chem_envs.append(envs)
        # print('ID:{} --- No. of adsorbates:{}'.format(i,len(envs)))
        for j,other_chems in enumerate(unique):
            if compare_chem_envs(envs,other_chems):
                # print('duplicate with {}'.format(unique_id[j]))
                Path("duplicate").mkdir(parents=True, exist_ok=True)
                path_dir = ('duplicate' + '/' + 'duplicate_with_{}'.format(j))
                make_fol_duplicate(j,path_dir,atoms)
                break
        else:
            Path("jobs").mkdir(parents=True, exist_ok=True)
            path_dir = ('jobs' + '/' + '{}'.format(i))
            make_fol(path_dir,atoms)
            unique.append(envs)
            unique_id.append(i)
            #system("cp -r" 'i' "unique/")
    # print('No of unique configurations are: {}'.format(len(unique_id)))
    # print(unique_id)
