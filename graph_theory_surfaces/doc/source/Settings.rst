==============================================
Surface Graphs: General Settings and Utilities
==============================================

The general settings can be divided in the following categories:
 * Graph generation settings
 * Graph analysis settings
 * Graph post-processing for different applications

*************************
Graph generation settings
*************************
These are settings that the user need to keep in mind, while attempting to generate a graph for a given system
 * Neigborlist: To generate the graph, the package requires a neighborlist. We currently use the neighborlist generated using the Atomic Simulation Environment (https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html). For the case of ase-neighborlist, the parameter of scaling the atomic radii and the skin factor are the required inputs. 
  
  + Atomic Radii Scaling: This number specifies how much the covalent radii of a given atom should be scaled. We have found a modest scaling of '1.1-1.2' to be sufficient for metal slabs.

  + skin factor: This is an input used as an error factor for all the bonds. The default setting of 0.25 Angstroms is sufficient.

 * Grid parameter: This parameter represents the number of periodic representations of the surface to make the surface graph. A grid setting of [1,1,0], will take '1' repetition of the unit cell in the +ve and -ve 'x' and 'y' directions respectively. The required grid size is dependent upon the size of the unit cell as well as on the chosen graph radius (described in the next bullet). Note however that large grids are undesirable as the scale poorly with the required computation, n representations in the grid increases the size of the graph by n^2.

 * Radius: This parameter determines the number of nearest neighbor shells that are to be considered in a given chemical environment. As our algorithm is geared towards catalysis, this parameter for the case of the graph constructed around a given adsorbate, will represent the number of neighboring surface shells that are desired by the user to capture the chemical environment of the adsorbate  

  + A radius of '0' represents only the adsorbate and the surface atoms that surround it. This radius, will help aid in finding the type of active site for the given adsorbate.

  + Radius of '1' represents a graph with the atoms encompassing the active site as well as the nearest neighbors of the active sites as well.

  + A 0.5 increament in the radii will capture any adsorbate atoms that are bound to surface nodes present in the graph. 

 * Node and Edge Properties: Graphs are highly versatile representations. An added benefit, which, makes them very useful for atomic scale simulations, are the node and edge attributes. For example, the distance between the surface atoms can be a useful edge property to store, or the coordination number of the surface site can be another important node property to store. These can be used to conviniently store important atomic information. A list of useful node and edge attributes is given below:
 
  + Node Properties:
 
    * Coordination number

    * D-band center

    * Electronegativity

    * number of valence electrons

  + Edge Properties:

    * Atom-atom bond distance

    * Type of bond: surface-surface, surface-ads, ads-ads, H--bond (some of this is captured in the "dist" parameter above)

    * bond energy of a given bond: this can be used to store entities like ads_energy between ads and surface atoms, vacancy formation energy between surface-surface atoms.

***********************
Graph analysis settings
***********************
After generating a full graph, and the subsequent chemical environments for all the adsorbate atoms, the generated graphs can be used in following ways:

* Unique chemical environments: For the case of complex adsorbate geometries such as high coverages of adsorbates and multi-dentate adsorbates, the graph representing the chemical environments of adsorbates can be used to systematically find unique chemical environments. Consider the case of 3 mono-dentate NO adsorbates on the surface. To estimate if this particular high coverage configuration is unique when compared to other 3 NO configurations, the chemical environments for each of the NO adsorbate in the case of interest, will be compared with the chemical environments of other cases having 3 NO. For the configuration to be unique, atleast one chemical environment of the three should be different for all the 3 NO cases in consideration. ** TODO: PUT A FIGURE TO EXPLAIN ** 

* Different settings for uniqueness: The most important setting for determining the uniqueness, with the above rule is the radius of the chemical environment considered. As explained in 'Graph generation settings' section above, the radius is the number of shells considered around the adsorbate. Therefore, larger the number of shells, the higher the number of interactions that are considered. Thus, the radius can in principle be large enough such as 4-5, such that for a given adsorbate, it captures all the atoms present on the surface model and can be used to estimate global minimum configurations, for that given unit cell. However, we have found that adsorption energies are not affected too much outside of 2-3 shells and that is the standard setting. 

************************************************
Graph post-processing for different applications
************************************************
This section explains some of the settings that we have found useful for different applications. 

* Evolutionary Algorithm: One way to systematically consider high coverages of intermediates is by using an evolutionary algorithm. The algorithm has a simple rule of only populating certain number of unique configurations for a particular coverage for the next high coverage configuration. The parameter of importance here, will be how many of the unique configurations are needed to be used to get a robust sampling to find global minimum for a given coverage. This is a case dependent parameter. In the case of strongly bound adsorbate, we found that considering configurations inside a small energy cutoff of 0.25 eV from the most stable case was sufficient. This selection was sufficient as the strong binding nature ensured that surface-adsorbate interaction was the dominant factor to consider, which helped screen next generation of coverages, using on a small subset of previous generation, which had all the relevant important interactions captured. However, in the case where there are competing interactions, such as surface-adsorbate interactions compete with adsorbate-adsorbate interactions, this choice of energy cutoff should be scrupoulously tested. One way to do this, is to plot energy histogram as a function of energy cutoff, this will aid in understanding the importance of certain kinds of interactions for a given system.    

* Setting to consider while considering high coverage configurations: Presented here is a list of settings, that we have found useful to systematically find high coverage configurations for different kinds of surface models:

  + Ads-Ads distance: This setting is very system dependent and depends on the nearest distance of two adjacent adsorption site. For example on Pt3Sn(111), this distance is of the order of 2.4 Angstroms, while on a Pt(100) surface, which is more open, a cutoff of about 2 Angstrom is sufficient, as the sites are inherently farther apart.

  + Explicitely constraining non-surface layers on non-terrace models: On non-terrace models like steps and kinks, we have found that the algorithm most efficiently works if the non-surface atoms (starting from the second layer) are constrained. This helps as the normals method used to find surface atoms, can fail in defected surfaces, as some sub-surface atoms can have non-zero normal components, especially post-relaxation.
