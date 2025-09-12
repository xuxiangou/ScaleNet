from torch.utils.data import Dataset
from graph.data import MGLDataset
from ace_gcn.generate_dataset import GraphData

class ScaleNetDataset(Dataset):
    def __init__(self,
                 threebody_cutoff,
                 structures,
                 graph_converter,
                 labels,
                 file_id_list,
                 include_line_graph=False,
                 return_struc=False):
        super(ScaleNetDataset, self).__init__()
        self.mgl_dataset = MGLDataset(
            threebody_cutoff=threebody_cutoff,
            structures=structures,
            converter=graph_converter,
            labels={"energy": labels},
            include_line_graph=include_line_graph,
        )
        self.cgcnn_dataset_1 = GraphData(
            structure_list=structures,
            label_list=labels,
            file_id_list=file_id_list,
            pickle_path="./data/OH_pickle_1",
            radius=0.8,
            grid=[1, 1, 1],
            return_structure=return_struc,
        )
        self.cgcnn_dataset_2 = GraphData(
            structure_list=structures,
            label_list=labels,
            file_id_list=file_id_list,
            pickle_path="./data/OH_pickle_2",
            radius=1.6,
            grid=[1, 1, 1],
            return_structure=return_struc,
        )

    def __len__(self):
        assert len(self.mgl_dataset) == len(self.cgcnn_dataset_1)
        return len(self.mgl_dataset)

    def __getitem__(self, idx):
        return *self.mgl_dataset[idx], *self.cgcnn_dataset_1[idx], *self.cgcnn_dataset_2[idx]
