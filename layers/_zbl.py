"""Zbl repulsive potential.The official implementation can be found in https://github.com/atomistic-machine-learning/schnetpack."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase.data import atomic_numbers

from layers._activations import softplus_inverse
from utils.cutoff import polynomial_cutoff


class NuclearRepulsion(nn.Module):
    """
    Ziegler-Biersack-Littmark style repulsion energy.

    Args:
        r_cut (float): cutoff for interaction range
        a0 (float): distance unit conversion from bohr into angstrom.
        ke (float): coulomb constant unit conversion from Ha into eV.
        trainable (bool):  whether the parameters are trainable.
    """

    def __init__(
        self,
        r_cut: float,
        a0: float = 0.5291772105638411,
        ke: float = 14.399645351950548,
        trainable: bool = False,
    ):
        super().__init__()
        self.r_cut = r_cut
        self.register_buffer("ke", torch.tensor(ke, dtype=torch.float32))

        a_div = softplus_inverse(torch.tensor([1.0 / (a0 * 0.8854)], dtype=torch.float32))
        a_pow = softplus_inverse(torch.tensor([0.23], dtype=torch.float32))
        exponents = softplus_inverse(torch.tensor([3.19980, 0.94229, 0.40290, 0.20162], dtype=torch.float32))
        coefficients = softplus_inverse(torch.tensor([0.18175, 0.50986, 0.28022, 0.02817], dtype=torch.float32))

        # Initialize network parameters
        self.a_pow = nn.Parameter(a_pow, requires_grad=trainable)
        self.a_div = nn.Parameter(a_div, requires_grad=trainable)
        self.coefficients = nn.Parameter(coefficients, requires_grad=trainable)
        self.exponents = nn.Parameter(exponents, requires_grad=trainable)

    def forward(self, element_types: tuple, graph: dgl.DGLGraph):
        """

        Args:
            element_types: A tuple of element types defined in the model class.
            graph: dgl.DGL graph.

        Returns:
            energy: Pairwise ZBL nuclear repulsive energy.
        """
        z_list = torch.tensor([atomic_numbers[i] for i in element_types], dtype=torch.float32)
        z = z_list[graph.ndata["node_type"]]

        idx_i = graph.edges()[0]
        idx_j = graph.edges()[1]

        r_ij = graph.edata["bond_dist"]

        # Construct screening function
        a = z ** F.softplus(self.a_pow)
        a_ij = (a[idx_i] + a[idx_j]) * F.softplus(self.a_div)
        # Get exponents and coefficients, normalize the latter
        exponents = a_ij[..., None] * F.softplus(self.exponents)[None, ...]
        coefficients = F.softplus(self.coefficients)[None, ...]
        coefficients = F.normalize(coefficients, p=1.0, dim=1)

        screening = torch.sum(coefficients * torch.exp(-exponents * r_ij[:, None]), dim=1)

        # Compute nuclear repulsion
        graph.edata["eij_repuls"] = (z[idx_i] * z[idx_j]) * polynomial_cutoff(r_ij, self.r_cut) * screening / r_ij

        graph.update_all(fn.copy_e("eij_repuls", "m_repuls"), fn.sum("m_repuls", "e_repuls"))

        energy = 0.5 * self.ke * dgl.readout_nodes(graph, "e_repuls", op="sum")

        return torch.squeeze(energy)
