from numpy.linalg import norm
from ase.neighborlist import NeighborList, natural_cutoffs
from itertools import combinations
from ase.constraints import constrained_indices

from surfgraph.chemical_environment import process_atoms
from surfgraph.chemical_environment import process_site
from surfgraph.chemical_environment import unique_chem_envs
from surfgraph.chemical_environment import bond_match
from surfgraph.helpers import draw_atomic_graphs
from surfgraph.helpers import normalize
from surfgraph.helpers import offset_position
from itertools  import combinations
import scipy
import networkx.algorithms.isomorphism as iso
import numpy as np
import networkx as nx


def plane_normal(xyz):
    """Return the surface normal vector to a plane of best fit. THIS CODE IS BORROWED FROM CATKIT

    Parameters
    ----------
    xyz : ndarray (n, 3)
        3D points to fit plane to.

    Returns
    -------
    vec : ndarray (1, 3)
        Unit vector normal to the plane of best fit.
    """
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]
    vec, _, _, _ = scipy.linalg.lstsq(A, xyz[:, 2])
    vec[2] = -1.0

    vec /= -np.linalg.norm(vec)

    return vec

def generate_normals_original(atoms, surface_normal=0.5, normalize_final=True, adsorbate_atoms=[]):
    normals = np.zeros(shape=(len(atoms), 3), dtype=float)

    atoms = atoms.copy()

    del atoms[adsorbate_atoms]

    cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    cell = atoms.get_cell()

    for index, atom in enumerate(atoms):
        normal = np.array([0, 0, 0], dtype=float)
        for neighbor, offset in zip(*nl.get_neighbors(index)):
            direction = atom.position - offset_position(atoms, neighbor, offset)
            normal += direction
        if norm(normal) > surface_normal:
            normals[index,:] = normalize(normal) if normalize_final else normal

    surface_mask = [index for index in range(len(atoms)) if norm(normals[index]) > 1e-5]

    return normals, surface_mask


def get_angle_cycle(a1,a2,a3):
    v1 = a1-a2
    v2 = a3-a2
    #print(v1,v2)
    nv1 = np.linalg.norm(v1)
    nv2 = np.linalg.norm(v2)
    if (nv1 <= 0).any() or (nv2 <= 0).any():
        raise ZeroDivisionError('Undefined angle')
    v1 /= nv1
    v2 /= nv2
    #print(v1,v2)
    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def generate_normals_new(atoms,cycle,nl,surface_mask):
    site_atoms = cycle.copy()
    if len(site_atoms) >2:
        atom_array = []
        for a in site_atoms:
            atom_array.append(atoms[a].position)
        normal = plane_normal(np.array(atom_array))

    else:
        neighbor_atoms = []
        for i in nl.get_neighbors(site_atoms[0])[0]:
            
            ### This if condition is to ensure that if the atoms are the same row, then the plane is formed btwn an atom in another row
            #print(i,atoms[i].position[2] - atoms[site_atoms[0]].position[2])
            if (i not in site_atoms) and i in nl.get_neighbors(site_atoms[1])[0] and i in surface_mask and i not in neighbor_atoms:
                neighbor_atoms.append(i)
        normal = [0, 0, 0]
        #print('Printing Neighboring atoms')
        #print(neighbor_atoms,site_atoms)
        for i in neighbor_atoms:
            site_atoms1 = site_atoms.copy()
            site_atoms1.append(i)
            atom_array = []
            initial = site_atoms[0]
            atom_array.append(atoms[initial].position)
            for a in site_atoms1:
                if a != initial:
                    a_offset = nl.get_neighbors(initial)[1][np.where(a==nl.get_neighbors(initial)[0])]
                    #print(a,np.dot(a_offset, atoms.get_cell())+atoms[a].position)
                    atom_array.append(atoms[a].position+np.dot(a_offset, atoms.get_cell())[0])
            if 0.85 < get_angle_cycle(atom_array[0],atom_array[1],atom_array[2]) < 1.2:
                normal += plane_normal(np.array(atom_array))
                #print('using this cycle to add to normal')
            #print(normal)
    normal = normalize(normal)
    return normal

def generate_site_type(atoms, surface_mask, normals, coordination, unallowed_elements=[]):
    cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    surface_mask = [index for index in surface_mask if atoms[index].symbol not in unallowed_elements]

    possible = list(combinations(set(surface_mask), coordination))
    valid = []
    sites = []

    for cycle in possible:
       for start, end in combinations(cycle, 2):
           if end not in nl.get_neighbors(start)[0]:
               break
       else: # All were valid
            valid.append(list(cycle))

    #print(valid)
    for cycle in valid:
        tracked = np.array(atoms[cycle[0]].position, dtype=float)
        known = np.zeros(shape=(coordination, 3), dtype=float)
        known[0] = tracked
        for index, (start, end) in enumerate(zip(cycle[:-1], cycle[1:])):
            for neighbor, offset in zip(*nl.get_neighbors(start)):
                if neighbor == end:
                    tracked += offset_position(atoms, neighbor, offset) - atoms[start].position
                    known[index + 1] = tracked

        average = np.average(known, axis=0)

        normal = np.zeros(3)

        #for index in cycle:
            #neighbors = len(nl.get_neighbors(index)[0])
            #normal += normals[index] * (1/neighbors)
        #normal = normalize(normal)
        #for index in cycle:
            #print(cycle)
        if len(cycle) == 1:
            for index in cycle:
                neighbors = len(nl.get_neighbors(index)[0])
                normal += normals[index] * (1/neighbors)
            normal = normalize(normal)

        if len(cycle) >1:
            neighbors = len(nl.get_neighbors(index)[0])
            cycle_orig = cycle
            #print(cycle)
            normal = generate_normals_new(atoms,cycle_orig,nl,surface_mask)
            #print(cycle,normal)
        for index in cycle:
            neighbors = len(nl.get_neighbors(index)[0])
            normal += normals[index] * (1/neighbors)
        normal = normalize(normal)

        if coordination ==2:
            average[2] = average[2] - 0.5
        if coordination == 3:
            average[2] = average[2] -0.7
            #print(average)
            #print(average[2])
        site_ads =Site(cycle=cycle, position=average, normal=normal)
        sites.append(site_ads)
        
    return sites

def generate_site_graphs(atoms, full_graph, nl, sites, adsorbate_atoms=[], radius=3):
    cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    site_envs = [None] * len(sites)
    for index, site in enumerate(sites):
        new_site = process_site(atoms, full_graph, nl, site.cycle, radius=radius)
        site_envs[index] = [new_site]
        site.graph = site_envs[index]

    unique_envs, unique_sites = unique_chem_envs(site_envs, sites)

    return unique_sites



def get_all_directed_graphs(K):
    DK=K.to_directed()
    edges_DK = list(DK.edges())
    permute_edges = combinations(edges_DK,len(K.edges()))
    #print(list(permute_edges))
    possible_Graph = []
    for edges in permute_edges:
        ads_G = nx.DiGraph()

        for i in range(0,len(edges)):
            ads_G_copy = ads_G.to_undirected()
            if not ads_G_copy.has_edge(edges[i][0],edges[i][1]):
                ads_G.add_edges_from([edges[i]])

        if len(ads_G.edges()) == len(K.edges()):
                possible_Graph.append(ads_G)

    return possible_Graph

def find_unique_directed_graphs(possible_Graph):
    unique_graph = []
    for i in possible_Graph:
        for j in unique_graph:
            if nx.is_isomorphic(i,j):
                break
        else:
            #print('adding')
            for x in i.nodes:
                if i.out_degree(x)>1:
                    break
            else:
                unique_graph.append(i)

    return unique_graph

def orient_H_bond(atoms):
    O_atoms = []
    ads_atoms = ['O','H','N']

    for i in atoms:
        if i.symbol == 'O':
            O_atoms.append(i.index)

    def find_H_index(atoms,O_index):
        H = []
        for i in atoms:
            if i.symbol =='H' and atoms.get_distance(O_index,i.index,mic=True) < 1.2:
                H.append(i.index)

        return H


    ## Find the O--O pairs that are separated by 3.2 Å, so that H--bond can be made between them
    OO_array = []
    for i in range(0,len(O_atoms)):
        for j in range(i+1,len(O_atoms)):
            if atoms.get_distance(O_atoms[i],O_atoms[j],mic =True) < 3.2 :
                OO_array.append([O_atoms[i],O_atoms[j]])

    ## Take all O-pairs and make an undirected graph
    K = nx.Graph()
    for i in OO_array:
        K.add_edges_from([(i[0],i[1])])

    ## Get all directed graphs from the undirected case and find all unique directed graphs.

    possible_graphs = get_all_directed_graphs(K)
    unique = find_unique_directed_graphs(possible_graphs)
    movie = []
    for i in unique:
        for n in i.nodes():
            H_top = find_H_index(atoms,n)
            atoms[H_top[0]].position = atoms[n].position + [0.3,0.3,0.9]
        atoms_edit = atoms.copy()
        #print(i.edges())
        for u,v in i.edges():
            vec=normalize(atoms.get_distance(u,v,mic=True,vector=True))
            #print('Tilting H attached to: {} '.format(u))
            H_atom = find_H_index(atoms,u)
            #print(H_atom)
            atom_H_orig = atoms[H_atom[0]].position
            atoms_edit[H_atom[0]].position = atoms_edit[u].position + vec
            H_top = find_H_index(atoms,v)
            if atoms_edit.get_distance(H_top[0],H_atom[0],mic = True) < 1.1:
                atoms_edit[H_top[0]].position = atoms_edit[v].position + [0.3,0.3,0.9]
            # Revert the position of the hydrogen back to the original position, if the H is really close
            #print(v,atoms.get_distance(H_atom[0],v,mic=True))
            if atoms_edit.get_distance(H_atom[0],v,mic=True) < 1.25:
                #print('reverting back H: {} because it is close to {} by {}'.format(H_atom[0],v,atoms_edit.get_distance(H_atom[0],v,mic=True)))
                atoms_edit[H_atom[0]].position = atom_H_orig
        movie.append(atoms_edit)



    return movie

class Site(object):
    def __init__(self, cycle, position, normal, graph=None):
        self.cycle = cycle
        self.position = position
        self.normal = normal
        self.graph = graph

    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, edge_match=bond_match)

    def __repr__(self):
        return "Cycle:{}, Position:[{}, {}, {}], Normal:[{}, {}, {}], Graph:{}".format(
                       self.cycle, 
                       self.position[0], self.position[1], self.position[2],
                       self.normal[0], self.normal[1], self.normal[2],
                       self.graph)    


    def adsorb(self, atoms, adsorbate, adsorbate_atoms, height=2, check_H_bond=False):

        ads_copy = adsorbate.copy()
        ads_atoms = adsorbate_atoms.copy()
        ads_copy.rotate([0, 0, 1], self.normal, center=[0,0,0])
        #print(self.position,self.position+ (self.normal*height), self.normal)
        ads_copy.translate(self.position + (self.normal*height))
        #print(len(atoms))
        atoms.extend(ads_copy)
        #print(len(atoms))
        index_to_check = range(len(atoms)-len(ads_copy), len(atoms))
        index_to_check_noH = []
        ads_atoms_check = []
        for ads_t in index_to_check:
            if atoms[ads_t].symbol != 'H':
                index_to_check_noH.append(ads_t)
        for ads_t in adsorbate_atoms:
            if atoms[ads_t].symbol != 'H':
                ads_atoms_check.append(ads_t)

        #print(index_to_check, index_to_check_noH)
        dist = float("inf")
        for ad in range(len(atoms)-len(ads_copy), len(atoms)):
            ads_atoms.append(ad)
        #print(ads_atoms)

        
        if len(adsorbate_atoms) != 0:
            for index in index_to_check_noH:
                dists = atoms.get_distances(index, ads_atoms_check, mic=True)
                dist = min(dist, dists.min())
        
        return dist

        
        
        
