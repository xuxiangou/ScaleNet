import random
from copy import deepcopy
from typing import List, Tuple, Optional, Union, Dict, Literal

from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, dict2constraint
import numpy as np
from ase import Atom, Atoms
from mp_api.client import MPRester
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.analysis.local_env import VoronoiNN, _handle_disorder
from pymatgen.core import Structure, Element
from pymatgen.core.surface import SlabGenerator, Slab
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from collections import OrderedDict

matter_proj_api_key = "IplIdHxX75IdF3s3LjLl3pqwKEeekXSy"


def get_matter_proj_connect():
    """
    Get the materials project connection

    :return: materials project connection object
    """
    matter_proj = MPRester(matter_proj_api_key)
    return matter_proj


def get_matter_structure_through_matter_id(matter_proj, matter_id):
    """
    Get the structure of material through material id using materials project api.
    If you just want to calculate the CN number, you don't need to set the conventional_unit_cell as true
    :param matter_proj: materials project object
    :param matter_id: material id of given material
    :return: material structure -> pymatgen.core Structure
    """
    # no need to obtain conventional unit cell because we do not calculate the adsorption energy,
    # just calculate the CN
    matter_structure = matter_proj.get_structure_by_material_id(material_id=matter_id,
                                                                final=True,
                                                                conventional_unit_cell=False)
    if type(matter_structure) == List:
        return matter_structure[0]  # we just want the first stable structure
    else:
        return matter_structure


def get_slabs_from_bulk_atoms(atoms, miller_indices: Tuple, make_super_cell=(2, 1, 1)):
    """
    Use pymatgen to enumerate the slabs from a bulk.
    :param atoms: The ase.Atoms object of the bulk that you want to make slabs out of
    :param miller_indices: A 3-tuple of integers containing the three Miller indices
                           of the slab[s] you want tomake.
    :return: A list of the slabs in the form of pymatgen.Structure
                objects. Note that there may be multiple slabs because
                of different shifts/terminations.
    """

    structure = Aseatoms_to_pymatgenstructure(atoms)
    sga = SpacegroupAnalyzer(structure, symprec=0.1)
    struct_stdrd = sga.get_conventional_standard_structure()
    slab_gen = SlabGenerator(initial_structure=struct_stdrd,
                             miller_index=miller_indices,
                             min_slab_size=8,
                             min_vacuum_size=15,
                             lll_reduce=False,
                             center_slab=False,
                             primitive=True,
                             )
    slabs = slab_gen.get_slabs(symmetrize=False, )
    slabs = [slab.make_supercell(list(make_super_cell))
             for slab in slabs]
    return slabs


def get_symmetry_CN(atoms):
    """
    Get a dictionary of coordination numbers
    for each distinct site in the bulk structure or slab structure.
    :param atoms: ase.Atoms or ase.Atom or pymatgen.core.Structure
    :return: Dict which presents the CN of the structure
             {'Ba2+': [11.122350151479433], 'Ti4+': [5.325488990556724], 'O2-': [5.325489011019475]}
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    sga = SpacegroupAnalyzer(structure)
    sym_structure = sga.get_symmetrized_structure()  # get a unique structure (unit cell)
    uni_indices = [equ[0] for equ in sym_structure.equivalent_indices]  # obtain the indices
    vor = VoronoiNN()
    cn_dict = OrderedDict()
    for idx in uni_indices:
        elem = sym_structure[idx].species_string
        if elem not in cn_dict.keys():
            cn_dict[elem] = []

        cn = vor.get_cn(sym_structure, idx, use_weights=True)
        if cn not in cn_dict[elem]:
            cn_dict[elem].append(cn)

    return cn_dict


on_disorder_options = Literal["take_majority_strict", "take_majority_drop", "take_max_species", "error"]


def get_cn(
        structure: Structure,
        n: int,
        use_weights: bool = False,
        on_disorder: on_disorder_options = "take_majority_strict",
) -> float:
    """
    Get coordination number, CN, of site with index n in structure.

    Args:
        structure (Structure): input structure.
        n (int): index of site for which to determine CN.
        use_weights (bool): flag indicating whether (True) to use weights for computing the coordination
            number or not (False, default: each coordinated site has equal weight).
        on_disorder ('take_majority_strict' | 'take_majority_drop' | 'take_max_species' | 'error'):
            What to do when encountering a disordered structure. 'error' will raise ValueError.
            'take_majority_strict' will use the majority specie on each site and raise
            ValueError if no majority exists. 'take_max_species' will use the first max specie
            on each site. For {{Fe: 0.4, O: 0.4, C: 0.2}}, 'error' and 'take_majority_strict'
            will raise ValueError, while 'take_majority_drop' ignores this site altogether and
            'take_max_species' will use Fe as the site specie.

    Returns:
        cn (int or float): coordination number.
    """
    vor = VoronoiNN()
    structure = _handle_disorder(structure, on_disorder)
    siw = vor.get_nn_info(structure, n)
    return sum(e["weight"] for e in siw) if use_weights else len(siw)


def get_all_CN(atoms,
               on_disorder: on_disorder_options = "take_majority_strict",
               use_weights: bool = True):
    """
    Get a dictionary of coordination numbers of all atoms (not dominating atoms)
    for each distinct site in the bulk structure or slab structure.
    :param atoms: ase.Atoms or ase.Atom or pymatgen.core.Structure
    :return: list which presents the CN of the structure
    """
    structure = _handle_disorder(Aseatoms_to_pymatgenstructure(atoms), on_disorder)
    vor = VoronoiNN()
    cn_list = []
    for idx, elem in enumerate(structure):
        siw = vor.get_nn_info(structure, idx)
        cn_list.append(sum(e["weight"] for e in siw) if use_weights else len(siw))

    return cn_list


# def get_adsorption_site_CN(atoms,
#                on_disorder: on_disorder_options = "take_majority_strict",
#                use_weights: bool = True):
#     """
#     Get a dictionary of coordination numbers of all atoms (not dominating atoms)
#     for each distinct site in the bulk structure or slab structure.
#     :param atoms: ase.Atoms or ase.Atom or pymatgen.core.Structure
#     :return: list which presents the CN of the structure
#     """
#     structure = _handle_disorder(Aseatoms_to_pymatgenstructure(atoms), on_disorder)
#     vor = VoronoiNN()
#     cn_list = []
#     for idx, elem in enumerate(structure):
#         siw = vor.get_nn_info(structure, idx)
#         cn_list.append(sum(e["weight"] for e in siw) if use_weights else len(siw))
#
#     return cn_list


def find_adsorption_sites(atoms) -> Dict:
    """
    A wrapper for pymatgen to get all the adsorption sites of a slab.

    :param atoms: The slab where you are trying to find adsorption sites in
                 `ase.Atoms` format

    :return: sites -> {"top": numpy.ndarray,
                       "bridge": numpy.ndarray,
                       "hollow_1": numpy.ndarray,
                       "hollow_2": numpy.ndarray}
    """

    structure = Aseatoms_to_pymatgenstructure(atoms)
    sites_dict = AdsorbateSiteFinder(structure).find_adsorption_sites(put_inside=True)
    top_sites = sites_dict["ontop"]
    bridge_sites = sites_dict["bridge"]
    hollow_sites = sites_dict["hollow"]
    return {
        "top": random.choice(top_sites),
        "bridge": random.choice(bridge_sites),
        "hollow_1": random.choice(hollow_sites),
        "hollow_2": random.choice(hollow_sites)
    }


def find_surface_atoms_indices(bulk_cn_dict, atoms):
    """
    A helper function referencing codes from pymatgen to
    get a list of surface atoms indices of a slab's
    top surface.

    :param bulk_cn_dict: A dictionary of coordination numbers
                        for each distinct site in the respective bulk structure
    :param atoms: The slab with substrate where you are trying to find surface sites in
                        `ase.Atoms` format

    :return:  indices_list(List[Turple]) -> A list that contains the indices of
                        the surface atoms
            examples: [30, 31, 32, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 57, 58]
    """

    structure = Aseatoms_to_pymatgenstructure(atoms)
    voronoi_nn = VoronoiNN()
    # Identify index of the surface atoms
    indices_list = []
    weights = [site.species.weight for site in structure]
    center_of_mass = np.average(structure.frac_coords,
                                weights=weights, axis=0)

    for idx, site in enumerate(structure):
        if site.frac_coords[2] > center_of_mass[2]:
            try:
                cn = voronoi_nn.get_cn(structure, idx, use_weights=True)
                cn = float('%.5f' % (round(cn, 5)))
                # surface atoms are under coordinated
                if cn < min(bulk_cn_dict[site.species_string]):
                    # slab do not contain the index info of atoms, the indices is used
                    indices_list.append(idx)
            except RuntimeError:
                # or if pathological error is returned,
                # indicating a surface site
                indices_list.append(idx)
    return indices_list


def get_atoms_electronegativity(atoms):
    """
    Get the electronegativity through ase.Atoms or pymatgen structure for all the atoms in the
    structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: electronegativity dict. Example: {'Ba': 0.89, 'Ti': 1.54, 'O': 3.44}
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    structure_with_no_oxidation = deepcopy(structure)
    structure_with_no_oxidation.remove_oxidation_states()
    electronegativeity_dict = {}
    for elem in structure_with_no_oxidation.elements:
        elem_electroneg = elem.X
        electronegativeity_dict[elem.symbol] = elem_electroneg if elem_electroneg != float("NaN") else 0.0

    return electronegativeity_dict


def get_atoms_mass(atoms):
    """
    Get the mass through ase.Atoms or pymatgen structure for all the atoms in the
    structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: atoms mass dict. Example: {"C": 12.0, "O": 14.0...}
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    structure_with_no_oxidation = deepcopy(structure)
    structure_with_no_oxidation.remove_oxidation_states()
    atom_mass_dict = {}
    for elem in structure_with_no_oxidation.elements:
        atom_mass_dict[elem.symbol] = Element(elem).atomic_mass

    return atom_mass_dict


def valence(self: Element):
    """From full electron config obtain valence subshell angular moment (L) and number of valence e- (v_e)."""
    if self.group == 18:
        return np.nan, 0  # The number of valence of noble gas is 0

    L_symbols = "SPDFGHIKLMNOQRTUVWXYZ"
    valence = []
    full_electron_config = self.full_electronic_structure
    last_orbital = full_electron_config[-1]
    for n, l_symbol, ne in full_electron_config:
        idx = L_symbols.lower().index(l_symbol)
        if ne < (2 * idx + 1) * 2 or (
            (n, l_symbol, ne) == last_orbital and ne == (2 * idx + 1) * 2 and len(valence) == 0
        ):  # check for full last shell (e.g. column 2)
            valence.append((idx, ne))
    # if len(valence) > 1:
    #     return valence

    return valence[0]


def get_adsorption_site(atoms, adsorption_atom_number: List):
    structure = Aseatoms_to_pymatgenstructure(atoms)
    adsorption_site_list = []
    for elem in structure:
        if elem.specie.number not in adsorption_atom_number:
            adsorption_site_list.append(0)
        else:
            adsorption_site_list.append(1)

    return adsorption_site_list


def get_valence_electron_number(atoms):
    """
    Get the valence electron number through ase.Atoms or pymatgen structure for all the atoms in the
    structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: valence electron number dict. Example: {'Ba': 2, 'Ti': 2, 'O': 4}
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    structure_with_no_oxidation = structure.copy()
    structure_with_no_oxidation.remove_oxidation_states()
    valence_electron_number_list = []
    for elem in structure_with_no_oxidation:
        valence_electron_number_list.append(valence(Element(elem.specie.symbol))[1])

    return valence_electron_number_list


def get_p_or_d_electron(atoms):
    """
    Get the p electrons number for p block elements and d electrons number for d block elements
    through ase.Atoms or pymatgen structure for all the atoms in the structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: p/d electrons number list in the structure order. Example: [1,3,6]
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    # structure_with_no_oxidation = deepcopy(structure)
    # structure_with_no_oxidation.remove_oxidation_states()
    p_d_electrons_number_list = []
    for elem in structure:
        elem_group = elem.specie.group
        electronic_structure = elem.specie.full_electronic_structure
        electronic_structure.reverse()
        if elem_group >= 13:
            for band in electronic_structure:
                if band[1] == "p":
                    p_d_electrons_number_list.append(int(band[-1]))
                    break

        elif 3 <= elem_group <= 12:
            for band in electronic_structure:
                if band[1] == "d":
                    p_d_electrons_number_list.append(int(band[-1]))
                    break
        else:
            p_d_electrons_number_list.append(0)
    return p_d_electrons_number_list


def get_electron_affinity(atoms):
    """
    Get the electron affinity number through ase.Atoms or pymatgen structure for all the atoms in the
    structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: electron affinity dict. Example: {'Ba': 0.144626, 'Ti': 0.075545, 'O': 1.4611053}
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    structure_with_no_oxidation = deepcopy(structure)
    structure_with_no_oxidation.remove_oxidation_states()
    electron_affinity_dict = {}
    for elem in structure_with_no_oxidation.elements:
        electron_affinity_dict[elem.symbol] = Element(elem).electron_affinity

    return electron_affinity_dict


def get_ionization_potential(atoms):
    """
    Get the ionization potential number through ase.Atoms or pymatgen structure for all
    the atoms in the structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: ionization energy dict. Example: {'Ba': 5.2116646, 'Ti': 6.82812, 'O': 13.618055}
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    structure_with_no_oxidation = deepcopy(structure)
    structure_with_no_oxidation.remove_oxidation_states()
    ionization_potential_dict = {}
    for elem in structure_with_no_oxidation.elements:
        ionization_potential_dict[elem.symbol] = Element(elem).ionization_energy

    return ionization_potential_dict


def get_interplanar_distance(atoms):
    """
    Get the interplanar distance through ase.Atoms or pymatgen structure for all
    the atoms in the structure.

    :param atoms: ase.Atoms or pymatgen.core.Structure

    :return: interplanar distance dict. Example: {(1, 0, 0): 4.933324312808996,
    (0, 1, 0): 4.933324312808997, (0, 0, 1): 7.006681, (1, 1, 0): 2.8482561200000003,
    (0, 1, 1): 4.033773419860285, (1, 1, 1): 2.638578725000418}

    """
    miller_index = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1)
    ]
    structure = Aseatoms_to_pymatgenstructure(atoms)
    tem_calc = TEMCalculator()
    interplanar_dis = tem_calc.get_interplanar_spacings(structure, miller_index)
    return interplanar_dis


def constrain_slab(atoms, z_cutoff=3.):
    """
    This function fixes sub-surface atoms of a slab. Also works on systems that
    have slabs + adsorbate(s), as long as the slab atoms are tagged with `0`
    and the adsorbate atoms are tagged with positive integers.

    :param atoms:   ASE-atoms class of the slab system. The tags of these atoms
                    must be set such that any slab atom is tagged with `0`, and
                    any adsorbate atom is tagged with a positive integer.
    :param z_cutoff:The threshold to see if slab atoms are in the same plane as
                    the highest atom in the slab.

    :return:    ase.Atoms(atoms) A deep copy of the `atoms` argument, but where the appropriate
                atoms are constrained
    """

    # Work on a copy so that we don't modify the original
    atoms = pymatgenstructure_to_Aseatoms(atoms)
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = []

    # If we assume that the third component of the unit cell lattice is
    # orthogonal to the slab surface, then atoms with higher values in the
    # third coordinate of their scaled positions are higher in the slab. We make
    # this assumption here, which means that we will be working with scaled
    # positions instead of Cartesian ones.
    scaled_positions = atoms.get_scaled_positions()
    unit_cell_height = np.linalg.norm(atoms.cell[2])

    # If the slab is pointing upwards, then fix atoms that are below the
    # threshold
    if atoms.cell[2, 2] > 0:
        max_height = max(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = max_height - z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] < threshold:
                mask.append(True)
            else:
                mask.append(False)

    # If the slab is pointing downwards, then fix atoms that are above the
    # threshold
    elif atoms.cell[2, 2] < 0:
        min_height = min(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = min_height + z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] > threshold:
                mask.append(True)
            else:
                mask.append(False)

    else:
        raise RuntimeError('Tried to constrain a slab that points in neither '
                           'the positive nor negative z directions, so we do '
                           'not know which side to fix')

    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms


def add_adsorbate_onto_slab(adsorbate: Atoms, slab: Atoms, site: Optional[Union[List, Tuple]]):
    """
    There are a lot of small details that need to be considered when adding an
    adsorbate onto a slab. This function will take care of those details for
    you.
    :param adsorbate: An `ase.Atoms` object of the adsorbate
    :param slab:  An `ase.Atoms` object of the slab
    :param site: A 3-long sequence containing floats that indicate the
                    cartesian coordinates of the site you want to add the
                    adsorbate onto.

    :return: adslab  An `ase.Atoms` object containing the slab and adsorbate.
                The sub-surface slab atoms will be fixed, and all adsorbate
                constraints should be preserved. Slab atoms will be tagged
                with a `0` and adsorbate atoms will be tagged with a `1`.
    """
    adsorbate = pymatgenstructure_to_Aseatoms(adsorbate)
    slab = pymatgenstructure_to_Aseatoms(slab)

    adsorbate = adsorbate.copy()  # To make sure we don't mess with the original
    adsorbate.translate(site)

    adslab = adsorbate + slab
    adslab.cell = slab.cell
    adslab.pbc = [True, True, True]

    # We set the tags of slab atoms to 0, and set the tags of the adsorbate to 1.
    # In future version of GASpy, we intend to set the tags of co-adsorbates
    # to 2, 3, 4... etc (per co-adsorbate)
    tags = [1] * len(adsorbate)
    tags.extend([0] * len(slab))
    adslab.set_tags(tags)

    # Fix the sub-surface atoms
    adslab_constrained = constrain_slab(adslab)

    return adslab_constrained


def get_surface_site_CN(slab_without_adsorbate, slab_with_adsorbate):
    """

    please note ❗ this method is only support one adsorbate !!!
    :param slab_without_adsorbate:
    :param slab_with_adsorbate:
    :return:
    """
    slab_without_adsorbate = Aseatoms_to_pymatgenstructure(slab_without_adsorbate)
    slab_with_adsorbate = Aseatoms_to_pymatgenstructure(slab_with_adsorbate)
    bulk_cn_symmetry_dict = get_symmetry_CN(slab_without_adsorbate)
    # Use slab to find rather than bulk!!! (in the bulk, all the CN for one atom is the same)
    without_adsorbate_surface_atoms_indices = find_surface_atoms_indices(bulk_cn_symmetry_dict,
                                                                         slab_without_adsorbate)
    if slab_with_adsorbate[0].species_string != slab_without_adsorbate[0].species_string:
        with_adsorbate_surface_atoms_indices = [idx + len(slab_with_adsorbate) - len(slab_without_adsorbate)
                                                for idx in without_adsorbate_surface_atoms_indices]
    else:
        with_adsorbate_surface_atoms_indices = without_adsorbate_surface_atoms_indices
    slab_with_adsorbate_all_atoms_CN = get_all_CN(slab_with_adsorbate)
    slab_with_adsorbate_surface_atoms_CN = [slab_with_adsorbate_all_atoms_CN[index] for index in
                                            with_adsorbate_surface_atoms_indices]

    return slab_with_adsorbate_surface_atoms_CN


def get_atoms_number(atoms):
    """
    Get the number of atoms of the pymatgen structure
    :param atoms: ase.Atoms or pymatgen.core Structure
    :return: int  number of atoms in this structure
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    return len(structure)


def get_atomic_number(atoms):
    """
    Get the atomic number of the pymatgen structure
    :param atoms: ase.Atoms or pymatgen.core Structure
    :return: List  atomic number of all the atoms of this structure
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    return list(structure.atomic_numbers)


def get_cart_coords(atoms):
    """
    Get the cartesian coordinate of all the atom of pymatgen structure
    :param atoms: ase.Atoms or pymatgen.core Structure
    :return: List shape(N, 3), cartesian coordinate
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    return list(structure.cart_coords)


def get_frac_coords(atoms):
    """
    Get the fractional coordinate of all the atom of pymatgen structure
    :param atoms: ase.Atoms or pymatgen.core Structure
    :return: List shape(N, 3), fractional coordinate
    """
    structure = Aseatoms_to_pymatgenstructure(atoms)
    return structure.frac_coords


def Aseatoms_to_pymatgenstructure(atoms):
    """
    Convert the ase.Atoms into pymatgen.core Structure...
    :param atoms -> ase.Atoms input
    :return: structure -> pymatgen.core Structure
    """
    if type(atoms) == Atoms or atoms.__class__.mro()[1] == Atoms:
        structure = AseAtomsAdaptor.get_structure(atoms)
    elif type(atoms) == Structure or atoms.__class__.mro()[1] == Structure:
        structure = atoms
    else:
        raise TypeError(f"Wrong input type, the program want to the Atoms or Structure but got {type(atoms)}")

    return structure


def pymatgenstructure_to_Aseatoms(structure):
    """
    Convert the pymatgen.core Structure into ase.Atoms...
    :param structure -> pymatgen.core Structure
    :return: atoms -> ase.Atoms input
    """
    if type(structure) == Structure or structure.__class__.mro()[1] == Structure:  # pymatgen.core Structure
        atoms = AseAtomsAdaptor.get_atoms(structure)
    elif type(structure) == Atoms or structure.__class__.mro()[1] == Atoms:
        atoms = structure
    else:
        raise TypeError(f"Wrong input type, the program want to the Atoms or Structure but got {type(structure)}")

    return atoms


def get_adsorbate():
    """
    Get all the adsorbate (you can add the extra one)
    :return: {str: ase.Atoms...}
    """
    adsorbate = {}
    adsorbate["H"] = Atoms("H", positions=[(0, 0, -0.5)])
    adsorbate["C"] = Atoms("C")
    adsorbate["O"] = Atoms("O")
    adsorbate["CO"] = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.1282]])

    return adsorbate


def get_atom_adsorb_type(doc):
    atoms: List = doc["atoms"]["atoms"]
    base_indices: List = doc["atoms"]["constraints"][0]["kwargs"]["indices"]
    adsorb_type_list: List = [0 for _ in range(doc["atoms"]["natoms"])]
    for idx, _ in enumerate(adsorb_type_list):
        if atoms[idx]["tag"] == 1:
            adsorb_type_list[idx] = 2
        elif idx in base_indices:
            adsorb_type_list[idx] = 0
        else:
            adsorb_type_list[idx] = 1
    return adsorb_type_list


def make_atoms_from_doc(doc):
    '''
    This is the inversion function for `make_doc_from_atoms`; it takes
    Mongo documents created by that function and turns them back into
    an ase.Atoms object.

    Args:
        doc     Dictionary/json/Mongo document created by the
                `make_doc_from_atoms` function.
    Returns:
        atoms   ase.Atoms object with an ase.SinglePointCalculator attached
    '''
    atoms = Atoms([Atom(atom['symbol'],
                        atom['position'],
                        momentum=atom['momentum'],
                        magmom=atom['magmom'],
                        charge=atom['charge'])
                   for atom in doc['atoms']['atoms']],
                  cell=doc['atoms']['cell'],
                  pbc=doc['atoms']['pbc'],
                  tags=get_atom_adsorb_type(doc),
                  info=doc['atoms']['info'],
                  constraint=[dict2constraint(constraint_dict)
                              for constraint_dict in doc['atoms']['constraints']])
    results = doc['results']
    calc = SinglePointCalculator(energy=results.get('energy', None),
                                 forces=results.get('forces', None),
                                 stress=results.get('stress', None),
                                 atoms=atoms)
    atoms.set_calculator(calc)
    return atoms


if __name__ == '__main__':
    from transforms import *

    # BaTiO3 = Structure.from_file("../../../pymatgen_tutorial/BaTiO3.cif", primitive=True)
    # print(BaTiO3)
    # sga = SpacegroupAnalyzer(BaTiO3)
    # css = sga.get_conventional_standard_structure()
    # with open('../../data/co_data.json', 'r') as file_handle:
    #     co_documents = json.load(file_handle)

    # s = Aseatoms_to_pymatgenstructure(make_atoms_from_doc(co_documents[1]))
    # rtf = RotationTransformation(axis=[1,0,0], angle=180)
    # s_new = rtf.apply_transformation(s)

    # CuO = Structure.from_file("../../../pymatgen_tutorial/structure/PtO2_mp-1285_computed.cif", primitive=False)
    # print(get_valence_electron_number(CuO))
    PtO = Structure.from_file("../data/O/200.cif")
    print(get_adsorption_site(PtO, [8]))

    # print(css)
    # print(BaTiO3.cart_coords)
    # dict = get_atoms_electronegativity(BaTiO3)
    # dict = get_valence_electron_number(BaTiO3)
    # dict = get_p_or_d_electron(BaTiO3)
    # dict = get_electron_affinity(BaTiO3)
    # dict = get_ionization_potential(BaTiO3)
    # dict = get_interplanar_distance(BaTiO3)
    # dict = get_surface_site_CN(BaTiO3)
    # dict = get_symmetry_CN(BaTiO3)
    # dict = get_all_CN(BaTiO3)
    # slabs = get_slabs_from_bulk_atoms(css, (1, 1, 1))[0]
    # matter_proj = get_matter_proj_connect()
    # print(get_matter_structure_through_matter_id(matter_proj, "mp-87"))
    # dict = find_surface_atoms_indices(BaTiO3)
    # print(BaTiO3.cart_coords)
    # rtf = RotationTransformation(axis=[1,0,0], angle=180)
    # tr_BaTiO3 = rtf.apply_transformation(BaTiO3)
    # print(tr_BaTiO3.cart_coords)
    # pst = PerturbStructureTransformation()
    # pst_BaTiO3 = pst.apply_transformation(BaTiO3)
    # print(pst_BaTiO3.cart_coords)
    # sat = SwapAxesTransformation()
    # sat_BaTiO3 = sat.apply_transformation(BaTiO3)
    # print(sat_BaTiO3.cart_coords)
    # print(slabs)
    # print(dict)
