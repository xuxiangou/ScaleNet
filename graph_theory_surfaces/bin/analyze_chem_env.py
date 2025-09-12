#!/usr/bin/env python
from ase.neighborlist import NeighborList, natural_cutoffs
import numpy as np
from surfgraph.chemical_environment import process_atoms
from surfgraph.chemical_environment import unique_chem_envs
from surfgraph.helpers import draw_atomic_graphs
from sys import argv
from ase.io import read
import argparse
import os
import shutil

parser = argparse.ArgumentParser(
    description='Some command line utilities for atomic graphs')
parser.add_argument(
    '--radius', '-radius', '-r',
    type=float, default=2,
    help='Sets the graph radius, this can be tuned for different behavior')
parser.add_argument(
    '--mult', '-mult', '-m',
    type=float, default=1.1,
    help='Sets the radii multiplier for the neighborlist')
parser.add_argument(
    '--skin', '-skin', '-s',
    type=float, default=0.25,
    help='Sets the skin for the neighborlist, \
          this is a constant offset extending a bond')
parser.add_argument(
    '--adsorbate-atoms', '-adsorbate-atoms', '-a',
    type=str, default='C,O,N,H',
    help='Comma delimited list of elements to be considered adsorbate (C,O,N,H default)')
parser.add_argument(
    '--grid', '-grid', '-g',
    type=str, default='2,2,0',
    help='Grid for PBC as comma delimited list, default is for surface (2,2,0 default)')
parser.add_argument(
    '--view-adsorbates', '-view-adsorbates',
    '--view', '-view', '-v',
    action='store_const',
    const=True, default=False,
    help='Displays the adsorbates using matplotlib')
parser.add_argument(
    '--unique', '-unique', '-u',
    action='store_const',
    const=True, default=False,
    help='Outputs the unique chemical environments')
parser.add_argument(
    '--direc', '-direc', '-d',
    action='store_const',
    const=True, default=False,
    help='Outputs the unique chemical environments in a directory')
parser.add_argument(
    'filenames',
    type=str,
    nargs='+',
    help='Atoms objects to process')
parser.add_argument(
    '--clean', '-clean','-c',
    type=str, default=None,
    help='Use clean atoms object')
args = parser.parse_args()

radii_multiplier = args.mult
adsorbate_elements = args.adsorbate_atoms.split(",")

all_atoms = []
filenames = []

for filename in args.filenames:
    try:
        all_atoms.append(read(filename))
        filenames.append(filename)
    except Exception as e:
        print("{} failed to read".format(filename))

chem_envs = []
energies = []
if args.clean:
    clean_atoms = read(args.clean)
    nl = NeighborList(natural_cutoffs(clean_atoms, radii_multiplier), self_interaction=False,
                      bothways=True, skin=args.skin)
    nl.update(clean_atoms)
    adsorbate_atoms = [atom.index for atom in clean_atoms if atom.symbol in adsorbate_elements]
    if len(adsorbate_atoms):
        raise Exception('Clean atoms should not have adsorbate atoms')
    args.clean, chem_env = process_atoms(clean_atoms, nl, adsorbate_atoms=adsorbate_atoms,
                                radius=args.radius, grid=[int(grid) for grid in args.grid.split(",")])   ## store the full graph for clean surface, to be used later
for atoms in all_atoms:
    try:
        energies.append(atoms.get_potential_energy())  ## Attempt to read the OUTCAR here
    except:
        energies.append(float('inf'))

    nl = NeighborList(natural_cutoffs(atoms, radii_multiplier), self_interaction=False,
                      bothways=True, skin=args.skin)
    nl.update(atoms)
    adsorbate_atoms = [atom.index for atom in atoms if atom.symbol in adsorbate_elements]
    full, chem_env = process_atoms(atoms, nl, adsorbate_atoms=adsorbate_atoms,
                                   radius=args.radius, grid=[int(grid) for grid in args.grid.split(",")],clean_graph=args.clean)  ## get the full graph and the chem_env for all the adsorbates found
    chem_envs.append(chem_env)

    labels = [None]*len(chem_env)
    for index, graph in enumerate(chem_env):
        labels[index] = {node:str(len([edge for edge in full[node] if edge.split(":")[0] not in adsorbate_elements])) for node in graph.nodes()}

    if args.view_adsorbates:
        print('Opening graph for viewing')
        draw_atomic_graphs(chem_env, atoms=atoms, labels=labels)


###### This next condition, finds the unique configs amongst the OUTCARS/ atoms object provided and arranges them according to ascending order of energies
if args.unique:
    unique, groups = unique_chem_envs(chem_envs, list(zip(energies, filenames)))
    print("Outputting the lowest energy unique configurations")
    groups = [sorted(group) for group in groups]
    for group in sorted(groups):
        if group[0][0] == float('inf'):
            print("{}: Unknown energy, {} duplicates".format(group[0][1], len(group) - 1))
        else:
            print("{}: {} eV, {} duplicates".format(group[0][1], group[0][0], len(group) - 1))
            if args.direc:
                if os.path.isdir('unique_dir'):
                    file_d = group[0][1].split('/')[0]
                    file_a = group[0][1].split('/')[1]
                    shutil.copyfile('./'+group[0][1],'./unique_dir/{}_{}'.format(file_a,file_d))
                else:
                    os.makedirs('./unique_dir')
                    file_d = group[0][1].split('/')[0]
                    file_a = group[0][1].split('/')[1]
                    shutil.copyfile(group[0][1],'./unique_dir/{}_{}'.format(file_a,file_d))
        for duplicate in group[1:]:
            if duplicate[0] == float('inf'):
                print("-> {}: Unknown energy".format(duplicate[1]))
            else:
                print("-> {}: {} eV".format(duplicate[1], duplicate[0]))
