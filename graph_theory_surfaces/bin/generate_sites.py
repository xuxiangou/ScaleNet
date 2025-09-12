#!/usr/bin/env python
from ase.visualize import view
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.constraints import constrained_indices
from ase import Atoms
from sys import argv
from glob import glob
from pathlib import Path
from numpy.linalg import norm

from surfgraph.chemical_environment import process_atoms
from surfgraph.site_detection import generate_normals_original as generate_normals
from surfgraph.site_detection import generate_site_type
from surfgraph.site_detection import generate_site_graphs
from surfgraph.site_detection import orient_H_bond

import argparse

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
    '--min-dist', '-min-dist',
    type=float, default=2,
    help='Sets the min distance between adsorbates that is allowed for new adsorbates')
parser.add_argument(
    '--adsorbate-atoms', '-adsorbate-atoms', '-a',
    type=str, default='C,O,N,H',
    help='Comma delimited list of elements to be considered adsorbate (C,O,N,H default)')
parser.add_argument(
    '--grid', '-grid', '-g',
    type=str, default='2,2,0',
    help='Grid for PBC as comma delimited list, default is for surface (2,2,0 default)')
parser.add_argument(
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
    '--no-adsorb', '-no-adsorb',
    type=str, default='',
    help='Comma delimited list of elements to not be considered for sites')
parser.add_argument(
    '--coordination', '-coordination', '-c',
    type=str, default='1,2,3',
    help='Coordinations to generate sites for as comma  delimited list, default is 1,2,3')
parser.add_argument(
    '--output', '-output', '-o',
    type=str, default=False,
    help='Outputs unique sites to files with given file extension')
parser.add_argument(
    '--output-dir', '-output-dir',
    type=str, default=".",
    help='Allows you to output files to a specific folder')
parser.add_argument(
    '--count-configs',
    action='store_const',
    const=True, default=False,
    help='Counts the number of configurations that are found')
parser.add_argument(
    'adsorbate',
    type=str,
    help='Adsorbate atoms object')
parser.add_argument(
    'filenames',
    type=str,
    nargs='+',
    help='Atoms objects to process')
args = parser.parse_args()

args.grid = [int(x) for x in args.grid.split(",")]
args.no_adsorb = args.no_adsorb.split(",")

movie = []
all_unique = []

for atoms_filename in args.filenames:
    atoms = read(atoms_filename)
    ads = read(args.adsorbate)
##########################################################################
    nl = NeighborList(natural_cutoffs(atoms, args.mult), self_interaction=False,  bothways=True, skin=args.skin)
    nl.update(atoms)

    adsorbate_atoms = [index for index, atom in enumerate(atoms) if atom.symbol in args.adsorbate_atoms]

    normals, mask = generate_normals(atoms,  surface_normal=0.5, adsorbate_atoms=adsorbate_atoms, normalize_final=True)   ### make sure to manually set the normals for 2-D materials, all atoms should have a normal pointing up, as all atoms are surface atoms
    #normals, mask = np.ones((len(atoms), 3)) * (0, 0, 1), list(range(len(atoms)))
    constrained = constrained_indices(atoms)
    mask = [index for index in mask if index not in constrained]
    #for index in mask:
    #    atoms[index].tag = 1

    atoms.set_velocities(normals/10)

    all_sites = []

    full_graph, envs = process_atoms(atoms, nl=nl, adsorbate_atoms=adsorbate_atoms, radius=args.radius, grid=args.grid) ### here the default radii as well as grid are considered, these can also be added as args.

    center = atoms.get_center_of_mass()

    for coord in [int(x) for x in args.coordination.split(",")]:
        found_count = 0
        found_sites = generate_site_type(atoms, mask, normals, coordination=coord, unallowed_elements=args.no_adsorb)

        for site in found_sites:
            all_sites.append(site)
        
        unique_sites = generate_site_graphs(atoms, full_graph, nl, found_sites, adsorbate_atoms=adsorbate_atoms, radius=args.radius)
        
        for index, sites in enumerate(unique_sites):
            new = atoms.copy()
            best_site = sites[0]
            
            for site in sites[1:]:
                if norm(site.position - center) < norm(best_site.position - center):
                    best_site = site
            #print(best_site.adsorb(new, ads, adsorbate_atoms),args.min_dist)
            ### this check is to ensure, that sites really close are not populated
            if best_site.adsorb(new, ads, adsorbate_atoms) >= args.min_dist:
                found_count += 1
                H_bond_movie = orient_H_bond(new)
                #print(H_bond_movie[:])
                if len(H_bond_movie) > 0:
                    for i in H_bond_movie:
                        movie.append(i)
                else:
                    movie.append(new)
                all_unique.append(site)
        if args.count_configs:
            print("{} coordination has {} configurations".format(coord, found_count))

    if args.view:
        view(movie)

if args.output:
    for index, atoms in enumerate(movie):
        atoms.write("./{}/{:05}.{}".format(args.output_dir, index, args.output))

