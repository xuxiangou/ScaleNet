import time

from ase.data.colors import jmol_colors
from ase.neighborlist import NeighborList, natural_cutoffs
from graph_theory_surfaces.surfgraph.helpers import grid_iterator
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import concurrent.futures
from functools import lru_cache
from joblib import Parallel, delayed

# Handles isomorphism for bonds
bond_match = iso.categorical_edge_match('bond', '')

# Handles isomorphism for atoms with regards to perodic boundary conditions
ads_match = iso.categorical_node_match(['index', 'ads'], [-1, False]) 

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def bond_symbol(atoms, a1, a2):
    return "{}{}".format(*sorted((atoms[a1].symbol, atoms[a2].symbol)))

def node_symbol(atom, offset):
    return "{}:{}[{},{},{}]".format(atom.symbol, atom.index, offset[0], offset[1], offset[2])

def add_atoms_node(graph, atoms, a1, o1, **kwargs):
    graph.add_node(node_symbol(atoms[a1], o1), index=a1, central_ads=False, **kwargs)

def add_atoms_edge(graph, atoms, a1, a2, o1, o2, adsorbate_atoms, **kwargs):
    dist = 2 - (1 if a1 in adsorbate_atoms else 0) - (1 if a2 in adsorbate_atoms else 0)

    graph.add_edge(node_symbol(atoms[a1], o1),
                   node_symbol(atoms[a2], o2),
                   bond=bond_symbol(atoms, a1, a2),
                   index='{}:{}'.format(*sorted([a1, a2])),
                   dist=dist,
                   dist_edge=atoms.get_distance(a1,a2,mic='True'),
                   ads_only=0 if (a1 in adsorbate_atoms and a2 in adsorbate_atoms) else 2,
                   **kwargs)

def compare_chem_envs(chem_envs1, chem_envs2):
    """Compares two sets of chemical environments to see if they are the same
    in chemical identity.  Useful for detecting if two sets of adsorbates are
    in the same configurations.
    
    Args:
        chem_envs1 (list[networkx.Graph]): A list of chemical environments, 
                                           including duplicate adsorbates
        chem_envs2 (list[networkx.Graph]): A list of chemical environments, 
                                           including duplicate adsorbates
    
    Returns:
        bool: Is there a matching graph (site / adsorbate) for each graph?
    """
    # Do they have the same number of adsorbates?
    if len(chem_envs1) != len(chem_envs2):
        return False

    envs_copy = chem_envs2[:] # Make copy of list

    # Check if chem_envs1 matches chem_envs2 by removing from envs_copy
    for env1 in chem_envs1: 
        for env2 in envs_copy:
            if nx.is_isomorphic(env1, env2, edge_match=bond_match):
                # Remove this from envs_copy and move onto next env in chem_envs1
                envs_copy.remove(env2)
                break

    # Everything should have been removed from envs_copy if everything had a match
    if len(envs_copy) > 0:
        return False

    return True

def unique_chem_envs(chem_envs_groups, metadata=None, verbose=False):
    """Given a list of chemical environments, find the unique
    environments and keep track of metadata if required.

    This function exists largely to help with unique site detection
    but its performance will scale badly with extremely large numbers
    of chemical environments to check.  This can be split into parallel
    jobs.

    Args:
        chem_env_groups (list[list[networkx.Graph]]): 
            Chemical environments to compare against each other
        metadata (list[object]): 
            Corresponding metadata to keep with each chemical environment

    Returns:
        list[list[list[networkx.Graph]]]: A list of unique chemical environments 
                                          with their duplicates
        list[list[object]]: A matching list of metadata
    """
    # Error checking, this should never really happen
    if len(chem_envs_groups) == 0:
        return [[],[]]

    # We have metadata to manage
    if metadata is not None:
        if len(chem_envs_groups) != len(metadata):
            raise ValueError("Metadata must be the same length as\
                              the number of chem_envs_groups")
    
    # No metadata to keep track of
    if metadata is None:
        metadata = [None] * len(chem_envs_groups)

    # Keep track of known unique environments
    unique = []

    for index, env in enumerate(chem_envs_groups):
        for index2, (unique_indices, unique_env) in enumerate(unique):
            if verbose:
                print("Checking for uniqueness {:05d}/{:05d} {:05d}/{:05d}".format(index+1, len(chem_envs_groups), index2, len(unique)), end='\r')
            if compare_chem_envs(env, unique_env):
                unique_indices.append(index)
                break
        else: # Was unique
            if verbose:
                print("")
            unique.append(([index], env))
    
    # Zip trick to split into two lists to return
    return zip(*[(env, [metadata[index] for index in indices]) for (indices, env) in unique])

def unique_adsorbates(chem_envs):
    """Removes duplicate adsorbates which occur when perodic
    boundary conditions detect the same adsorbate in two places. Each
    adsorbate graph has its atomic index checked to detect when PBC has 
    created duplicates.

    Args:
        chem_env_groups (list[networkx.Graph]): Adsorbate/environment graphs

    Returns:
        list[networkx.Graph]: The unique adsorbate graphs
    """
    # Keep track of known unique adsorbate graphs
    unique = []
    for env in chem_envs:
        for unique_env in unique:
            if nx.is_isomorphic(env, unique_env, edge_match=bond_match, node_match=ads_match):
                break
        else: # Was unique
            unique.append(env)
    return unique


def process_site(atoms, full, nl, site, radius=3):
    #neighbors = nl.neighbors
    #offsets = nl.displacements
    #neighbors, offsets = nl.get_neighbors()
    full.add_node("X", index=None)
    offset = np.array([0, 0, 0])
    full.add_edge("X",
                  node_symbol(atoms[site[0]], offset),
                  bond="X:{}".format(atoms[site[0]].symbol),
                  ads=0) # Handle first manually
    for last_index, next_index in zip(site[:-1], site[1:]):
        # Error handling needed, .index could be None / -1?
        neighbors, offsets = nl.get_neighbors(last_index)
        #neighbor_index = list(neighbors[last_index]).index(next_index)
        neighbor_index = list(neighbors).index(next_index)
        #offset += offsets[last_index][neighbor_index]
        offset += offsets[neighbor_index]
        #print(offset)
        full.add_edge("X",
                      node_symbol(atoms[next_index], offset),
                      bond="X:{}".format(atoms[next_index].symbol),
                      ads=0)

    site_graph = nx.ego_graph(full, "X", radius=(radius*2)+1, distance="dist")
    site_graph = nx.subgraph(full, list(site_graph.nodes()))
    site_graph = nx.Graph(site_graph)
    full.remove_node("X")
    return site_graph

def check_H_bond_angle(H_index, O_index, atoms,adsorbate_atoms):
    for index in adsorbate_atoms:
        if atoms[int(index)].symbol == 'O' and (atoms.get_distance(index,H_index,mic=True)<1.2):
            # print('The donor H is attached to a Oxygen:{}'.format(index))
            donor_O = int(index)
            angle = atoms.get_angle(donor_O,H_index,O_index,mic=True)
            # print('the found angle is {}'.format(abs(180-abs(angle))))
            if abs(180-abs(angle)) < 80:   #### set some tolerance for how much the bond is allowed to deviate
                return True
            else:
                return False

def H2O_angle_check(H2O_array,atoms):
    angle = atoms.get_angle(H2O_array[1],H2O_array[0],H2O_array[2],mic=True)
    # print('the found angle is {}'.format(abs(angle)))
    if 90<abs(angle)<120:
        return True
    else:
        return False

def find_H_bonds(atoms, adsorbate_atoms):
    O_H_bonded = []
    H_O_bonded = []
    H2O_mol = []
    OH_len = []
    for index in adsorbate_atoms:
        H2O_array = []

        if atoms[int(index)].symbol == 'O':
            H2O_array.append(index)
            for index_2 in adsorbate_atoms:
                #print(index,index_2)
                if atoms[int(index_2)].symbol == 'H' and (1.25 < atoms.get_distance(index,index_2,mic=True) < 2.1):
                    #print(index,index_2)
                    if check_H_bond_angle(int(index_2),int(index),atoms,adsorbate_atoms):
                        O_H_bonded.append(index)
                        H_O_bonded.append(index_2)
                        OH_len.append( atoms.get_distance(index,index_2,mic=True))
                if atoms[int(index_2)].symbol == 'H' and (atoms.get_distance(index,index_2,mic=True) < 1.2):
                    #print('adding {} to H2O array'.format(index_2))
                    H2O_array.append(index_2)

            if len(H2O_array) == 3 and H2O_angle_check(H2O_array,atoms):
                #print(H2O_array)
                H2O_mol.append(index)
    return O_H_bonded, H_O_bonded, OH_len, H2O_mol

@lru_cache(maxsize=None)
def cached_get_neighbors(nl, index):
    return nl.get_neighbors(index)

# 处理单个原子的邻居和网格信息，支持并行化
def process_per_atom(nl,distances,grid, atom_info):
    grid_x_min, grid_x_max = -grid[0], grid[0]
    grid_y_min, grid_y_max = -grid[1], grid[1]
    grid_z_min, grid_z_max = -grid[2], grid[2]
    index, atom, grid, adsorbate_atoms = atom_info
    neighbors, offsets = cached_get_neighbors(nl, index)

    # 使用 NumPy 进行矢量化处理，过滤有效的邻居
    neighbor_distances = np.array([distances[index][neighbor] for neighbor in neighbors])
    adsorbate_mask = np.array(
        [(index in adsorbate_atoms) != (neighbor in adsorbate_atoms) for neighbor in neighbors])

    # 过滤出满足条件的邻居
    valid_mask = (neighbor_distances <= 2.6) | ((neighbor_distances > 2.6) & adsorbate_mask)
    valid_neighbors = [(neighbors[i], offsets[i]) for i in np.where(valid_mask)[0]]

    results = []
    for x, y, z in grid_iterator(grid):
        for neighbor, offset in valid_neighbors:
            ox, oy, oz = offset
            # 优化：合并坐标条件检查
            if grid_x_min <= ox + x <= grid_x_max and grid_y_min <= oy + y <= grid_y_max and grid_z_min <= oz + z <= grid_z_max:
                results.append((index, neighbor, (x, y, z), (x + ox, y + oy, z + oz)))

    return results

def process_atoms(atoms, nl, adsorbate_atoms=None, radius=2, grid=(2, 2, 0), clean_graph=None, H_bond=False):
    """Takes an ase Atoms object and processes it into a full graph as well as
    a list of adsorbate graphs.  This allows for the further post processing
    of the graph to identify chemical environments, chemical identity, and
    site detection.

    Args:
        atoms (ase.Atoms): The atoms object to process
        nl (ase.neighborlist.Neighborlist): A neighborlist built on the atoms object
        adsorbate_atoms (list[int]): The indices of adsorbate atoms
        radius (int): The radius for adsorbate graphs, this is a tunable parameter
        grid (tuple[int]): (X,Y,Z) repetitions for PBC.  This should be raised until
                           graphs do not form loops across PBC.  A good starting point
                           for surface DFT is (2,2,0) but this may change if radius is
                           large.

    Returns:
        networkx.Graph: The full graph containing every atom
        list[networkx.Graph]: A list of adsorbate graphs without duplicates from perodic
                              boundary conditions.
    """
    distances = atoms.get_all_distances(mic=True)

    full = nx.Graph()
    # Grid argument determines how many edge repetitions are added.  Scaling for this will be extremely bad
    # (2 * (n - 1) + 1) ^ 2 scaling

    # Add all atoms to graph
    for index, atom in enumerate(atoms):
        for x, y, z in grid_iterator(grid):
            add_atoms_node(full, atoms, index, (x, y, z))   

    if H_bond == True:
        O_H_bonded,H_O_bonded,OH_len,H2O_mol = find_H_bonds(atoms, adsorbate_atoms)
        # print('Printing relevant information')
        # print(O_H_bonded, OH_len, H2O_mol)
        for ind,i in enumerate(full.nodes(data=True)):
            Hb_array = []
            #print(H_O_bonded)
            if i[1]['index'] in H_O_bonded:
                O_result =  H_O_bonded.index(i[1]['index'])
                full.nodes[str(list(full.nodes)[ind])]['H_bond_O'] = O_H_bonded[O_result]
                full.nodes[str(list(full.nodes)[ind])]['H_bond_len'] = OH_len[O_result]
            if i[1]['index'] in O_H_bonded:
                H_bonds = O_H_bonded.count(i[1]['index'])
                #print(int(i[1]['index']),O_H_bonded)
                H_result = np.where(np.array(O_H_bonded) == i[1]['index'])
                for hbi in H_result[0]:
                    #print(hbi,OH_len[hbi])
                    Hb_array.append(OH_len[hbi])
                #Hb_array.append(OH_len[hbi] for hbi in H_result[0])
                #print(H_result,Hb_array)
                #print(H_bonds)
                #print(full.nodes[str(list(full.nodes)[ind])])
                full.nodes[str(list(full.nodes)[ind])]['H_bond'] = H_bonds
                full.nodes[str(list(full.nodes)[ind])]['H_bond_len'] = Hb_array
            if i[1]['index'] in H2O_mol:
                H_bonds = H2O_mol.count(i[1]['index'])
                #print(H_bonds)
                #print(full.nodes[str(list(full.nodes)[ind])])
                full.nodes[str(list(full.nodes)[ind])]['H2O_mol'] = H_bonds
    # Add all edges to graph
    # a = time.time()
    # for index, atom in enumerate(atoms):
    #     for x, y, z in grid_iterator(grid):
    #         neighbors, offsets = nl.get_neighbors(index)
    #         for neighbor, offset in zip(neighbors, offsets):
    #             ox, oy, oz = offset
    #             if not (-grid[0] <= ox + x <= grid[0]):
    #                 continue
    #             if not (-grid[1] <= oy + y <= grid[1]):
    #                 continue
    #             if not (-grid[2] <= oz + z <= grid[2]):
    #                 continue
    #             # This line ensures that only surface adsorbate bonds are accounted for that are less than 2.5 Å
    #             if distances[index][neighbor] > 2.6 and (bool(index in adsorbate_atoms) ^ bool(neighbor in adsorbate_atoms)):
    #                 continue
    #             add_atoms_edge(full, atoms, index, neighbor, (x, y, z), (x + ox, y + oy, z + oz), adsorbate_atoms)

    # 优化：将 adsorbate_atoms 转换为集合以提高查找速度
    adsorbate_atoms = set(adsorbate_atoms)

    # 优化：并行化处理每个原子及其邻居
    atom_info_list = [(index, atom, grid, adsorbate_atoms) for index, atom in enumerate(atoms)]
    # 选择并行处理的方式，可以根据任务的 CPU 密集度选择 ThreadPoolExecutor 或 multiprocessing.Pool
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     results = executor.map(process_atom, atom_info_list)
    atoms_results = Parallel(n_jobs=1)(
        delayed(process_per_atom)(nl, distances,grid, atom_info) for atom_info in atom_info_list
    )
    # 将并行化后的结果汇总并处理
    for result in atoms_results:
        if len(result) != 0:
            for edge_data in result:
                add_atoms_edge(full, atoms, *edge_data, adsorbate_atoms)

    # for result in atoms_results:
    #     if len(result) != 0:
    #         for edge_data in result:
    #             add_atoms_edge(full, atoms, *edge_data, adsorbate_atoms)
    ###########################################################################
    # Add the case of surface-ads + ads-ads edges to clean graph case here
    if clean_graph:
        edges = [(u, v, d) for u, v,d in full.edges.data() if d["dist"] < 2]    ### Read all the edges, that are between adsorbate and surface (dist<2 condition)
        nodes = [(n, d) for n, d in full.nodes.data() if d["index"] in adsorbate_atoms]    ### take all the nodes that have an adsorbate atoms
        full=nx.Graph(clean_graph)
        full.add_nodes_from(nodes)
        full.add_edges_from(edges)
        
    # All adsorbates into single graph, no surface
    ads_nodes = None
    ads_nodes = [node_symbol(atoms[index], (0, 0, 0)) for index in adsorbate_atoms]
    ads_graphs = nx.subgraph(full, ads_nodes)

    # Cut apart graph
    ads_graphs = connected_component_subgraphs(ads_graphs) 

    chem_envs = []
    for ads in ads_graphs:
        initial = list(ads.nodes())[0]
        full_ads = nx.ego_graph(full, initial, radius=0, distance="ads_only")

        new_ads = nx.ego_graph(full, initial, radius=(radius*2)+1, distance="dist")
        new_ads = nx.Graph(nx.subgraph(full, list(new_ads.nodes())))
        for node in ads.nodes():
            new_ads.add_node(node, central_ads=True)

        for node in full_ads.nodes():
            new_ads.add_node(node, ads=True)
        
        chem_envs.append(new_ads)

    chem_envs = unique_adsorbates(chem_envs)  

    chem_envs.sort(key=lambda x: len(x.edges()))

    return full, chem_envs
