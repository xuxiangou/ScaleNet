from typing import Literal
import torch
import torch.nn as nn
from tensornet import TensorNet
from ace_gcn.ace_gcn_model import CrystalGraphConvNet
from layers._core import MLP
from layers._activations import ActivationFunction


class ScaleNet(nn.Module):
    def __init__(self,
                 elem_list,
                 units,
                 cutoff,
                 rbf_type,
                 nblocks,
                 use_smooth,
                 is_intensive,
                 field,
                 orig_atom_fea_len,
                 nbr_fea_dist_len,
                 nbr_cat_value,
                 activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
                 ):
        super(ScaleNet, self).__init__()
        self.tensornet = TensorNet(
            element_types=elem_list,
            units=units,
            cutoff=cutoff,
            rbf_type=rbf_type,
            nblocks=nblocks,
            use_smooth=use_smooth,
            is_intensive=is_intensive,
            field=field,
        )

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None
        self.cgcnn = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_dist_len, nbr_cat_value)
        # dims_final_layer = [2 * units, units, units, units]
        # self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
        # self.mlp = MLP([2 * units, units], activation, activate_last=True)
        # self.mlp_2 = MLP([2 * units, units], activation, activate_last=True)
        # self.final = MLP([units, 1], activation, activate_last=False)
        self.mlp = nn.Linear(2 * units, units)
        self.mlp_2 = nn.Linear(2 * units, units)
        # self.mlp_3 = nn.Linear(2 * units, units)
        self.final = nn.Linear(units, 1)

    def forward(self, g, state_attr, inputs, inputs_2):
        adsorbate_info, nanoparticle_info = self.tensornet(g, state_attr)
        # local_info = self.cgcnn(*inputs)
        local_info_2 = self.cgcnn(*inputs_2)
        m = torch.cat((adsorbate_info, local_info_2), dim=-1)
        m = self.mlp(m)
        # m = torch.cat((adsorbate_info, m), dim=-1)
        # m = torch.cat((adsorbate_info, local_info_2), dim=-1)
        # m = global_info + local_info
        # m = self.mlp_2(m)
        m = torch.cat((nanoparticle_info, m), dim=-1)
        m = self.mlp_2(m)
        return self.final(m)
