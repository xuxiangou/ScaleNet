import copy
from copy import deepcopy
from functools import partial
import pandas as pd
import os.path
from pathlib import Path
import random
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data.utils import Subset
from logzero import logger, logfile
from pymatgen.core import Structure
from ruamel.yaml import YAML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from ScaleNetDataset import ScaleNetDataset
from train_utils.utils import save_model, load_model, plot_model_metric, plot_comparison_pic, create_dir
from train_utils.schedules import LinearWarmupExponentialDecay
from ext.pymatgen import Structure2Graph, get_element_list
from graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from scaleNet import ScaleNet
from ace_gcn.ace_gcn_model import CrystalGraphConvNet
from pymatgen.io.cif import CifParser

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
dgl.seed(seed)

class Normalizer():
    def __init__(self, mean, std):
        self.mean = mean.clone().detach()
        self.std = std.clone().detach()

    @torch.no_grad()
    def norm(self, x):
        return (x - self.mean) / self.std

    @torch.no_grad()
    def denorm(self, x):
        return (x * self.std) + self.mean

def load_data(data_dir):
    label_file = os.path.join(data_dir, "id_prop.csv")
    label_dict = pd.read_csv(label_file, header=None, index_col=0).to_dict()[1]
    file_id = []
    label_list = []
    structure_list = []
    for file_name, label in label_dict.items():
        structure_file_dir = os.path.join(data_dir, str(file_name) + ".cif")
        structure = CifParser(structure_file_dir, frac_tolerance=0).parse_structures(primitive=False)[0]
        file_id.append(file_name)
        structure_list.append(structure)
        label_list.append(float(label))
    return structure_list, label_list, file_id

def scale_net_collate_fn(batches, multiple_values_per_target: bool = False):
    graphs = [batch[0] for batch in batches]
    lattices = [batch[1][0] for batch in batches]
    state_attr = [batch[2] for batch in batches]
    labels = [list(batch[3].values())[0] for batch in batches]
    g = dgl.batch(graphs)
    lat = torch.stack(lattices)
    labels = torch.stack(labels)
    state_attr = torch.stack(state_attr)

    def convert_cgcnn_format(batch):
        batch_atom_fea, batch_nbr_dist_fea, batch_nbr_adj, batch_nbr_bond_type = [], [], [], []
        batch_self_fea_idx, batch_nbr_fea_idx = [], []
        crystal_atom_idx, batch_target = [], []
        ads_atom_idx = []
        batch_outcar_ids = []
        base_idx = 0

        for i, ((total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat,
                 total_self_fea_idx, total_nbr_index_node), target, outcar_id) in enumerate(batch):
            num_ads = len(total_node_list)
            total_nodes = sum([len(total_node_list[i]) for i in range(0, num_ads)])

            for ads_iter in range(0, num_ads):
                n_i = len(total_node_list[ads_iter])
                batch_atom_fea.append(total_atom_fea[ads_iter])
                batch_nbr_dist_fea.append(total_nbr_dist_node[ads_iter])
                batch_nbr_adj.append(total_nbr_adj_node[ads_iter])
                batch_nbr_bond_type.append(total_nbr_bond_cat[ads_iter])
                batch_self_fea_idx.extend([total_self_fea_idx[ads_iter] + base_idx])
                batch_nbr_fea_idx.extend([total_nbr_index_node[ads_iter] + base_idx])
                ads_atom_idx.extend([ads_iter + base_idx] * len(total_node_list[ads_iter]))
                base_idx += n_i

            crystal_atom_idx.extend([i] * total_nodes)
            batch_target.append(target)
            batch_outcar_ids.append(outcar_id)

        return ((torch.cat(batch_atom_fea, dim=0),
                    torch.cat(batch_nbr_dist_fea, dim=0),
                    torch.cat(batch_nbr_adj, dim=0),
                    torch.cat(batch_nbr_bond_type, dim=0),
                    torch.cat(batch_self_fea_idx, dim=0),
                    torch.cat(batch_nbr_fea_idx, dim=0),
                    torch.LongTensor(ads_atom_idx),
                    torch.LongTensor(crystal_atom_idx)),
                torch.stack(batch_target, dim=0),batch_outcar_ids)

    return (g,
            lat,
            state_attr,
            labels,
            *convert_cgcnn_format([batch[4:7] for batch in batches]),
            *convert_cgcnn_format([batch[7:] for batch in batches]))

class Trainer():
    def __init__(self):
        super().__init__()

    @staticmethod
    def split_dataset(dataset, num_train=0.8, num_valid=0.1, shuffle=False, random_state=None):
        from itertools import accumulate
        num_data = len(dataset)
        assert num_train + num_valid <= 1.0
        lengths = [int(num_data * num_train), int(num_data * num_valid),
                   num_data - int(num_data * num_train) - int(num_data * num_valid)]
        if shuffle:
            indices = np.random.RandomState(seed=random_state).permutation(num_data)
        else:
            indices = np.arange(num_data)

        data_dict = [
            Subset(dataset, indices[offset - length: offset])
            for offset, length in zip(accumulate(lengths), lengths)
        ]
        return data_dict

    @torch.no_grad()
    def ema(self, ema_model, model, decay):
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)

    def edge_init(self, edges):
        R_src, R_dst = edges.src["R"], edges.dst["R"]
        dist = torch.sqrt(F.relu(torch.sum((R_src - R_dst) ** 2, -1)))
        return {"d": dist, "o": R_src - R_dst}

    def _collate_fn(self, batch):
        graphs, line_graphs, labels, *args = map(list, zip(*batch))
        g, l_g = dgl.batch(graphs), dgl.batch(line_graphs)
        labels = torch.tensor(labels, dtype=torch.float32)
        return g, l_g, labels

    def train(self, device, model, opt, loss_fn, grad_clip, train_loader, scaler, normalizer):
        model.train()
        epoch_loss = 0
        num_samples = 0

        for g, lat, state_attr, labels, inputs, _, _, inputs_2, _, _ in train_loader:
            opt.zero_grad()
            g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
            g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            g.ndata["pos"] = (
                    g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
            ).sum(dim=1)
            g = g.to(device)
            state_attr = state_attr.to(device)
            inputs = (input.to(device) for input in inputs)
            inputs_2 = (input.to(device) for input in inputs_2)
            labels = normalizer.norm(labels).to(device)
            logits = model(g, state_attr, inputs, inputs_2)
            loss = loss_fn(logits, labels.view(logits.shape[0], -1))
            epoch_loss += loss.data.item() * len(labels)
            num_samples += len(labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=grad_clip)
            opt.step()

        return epoch_loss / num_samples

    @torch.no_grad()
    def evaluate(self, device, model, valid_loader, normalizer):
        model.eval()
        predictions_all, labels_all = [], []

        for g, lat, state_attr, labels, inputs, _, _, inputs_2, _, _ in valid_loader:
            g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
            g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            g.ndata["pos"] = (
                    g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
            ).sum(dim=1)
            g = g.to(device)
            state_attr = state_attr.to(device)
            inputs = (input.to(device) for input in inputs)
            inputs_2 = (input.to(device) for input in inputs_2)
            logits = model(g, state_attr, inputs, inputs_2)
            labels_all.extend(labels)
            predictions_all.extend(
                normalizer.denorm(logits.view(-1).cpu()).numpy()
            )

        return np.array(predictions_all), np.array(labels_all)

    def main(self, model_cnf_dir):
        yaml = YAML(typ="safe")
        model_cnf = yaml.load(Path(model_cnf_dir))
        model_name, model_params, train_params, pretrain_params, data_params = (
            model_cnf["name"],
            model_cnf["model"],
            model_cnf["train"],
            model_cnf["pretrain"],
            model_cnf["data"]
        )
        if not os.path.exists(model_params["model_save_dir"]):
            os.mkdir(model_params["model_save_dir"])
        model_save_dir = create_dir(model_params["model_save_dir"])
        logfile(os.path.join(model_save_dir, "run.log"))

        logger.info(f"Model name: {model_name}")
        logger.info(f"Model params: {model_params}")
        logger.info(f"Train params: {train_params}")
        logger.info(f"Data params: {data_params}")

        logger.info("Loading Data Set")
        structure_list, label_list, file_name_list = load_data(data_params["raw_data_dir"])
        elem_list = get_element_list(structure_list)
        converter = Structure2Graph(element_types=elem_list, cutoff=model_params["cutoff"])
        scale_net_dataset = ScaleNetDataset(
            threebody_cutoff=model_params["cutoff"],
            structures=structure_list,
            graph_converter=converter,
            labels=label_list,
            file_id_list=file_name_list,
            include_line_graph=False,
        )

        # Check if cross-validation is enabled
        n_folds = data_params.get("n_folds", 1)
        if n_folds > 1:
            logger.info(f"Starting {n_folds}-fold cross-validation")
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_metrics = []

            for fold, (train_valid_idx, test_idx) in enumerate(kf.split(scale_net_dataset)):
                logger.info(f"Fold {fold + 1}/{n_folds}")
                fold_save_dir = os.path.join(model_save_dir, f"fold_{fold+1}")
                os.mkdir(fold_save_dir)
                logfile(os.path.join(fold_save_dir, "run.log"))

                # Split train_valid into train and validation
                train_valid_set = Subset(scale_net_dataset, train_valid_idx)
                num_train = int(len(train_valid_set) * train_params["num_train"])
                num_valid = len(train_valid_set) - num_train
                train_set, valid_set = torch.utils.data.random_split(
                    train_valid_set, [num_train, num_valid], 
                    generator=torch.Generator().manual_seed(seed)
                )
                test_set = Subset(scale_net_dataset, test_idx)

                # Compute normalizer from training set
                train_labels = [scale_net_dataset[i][3]["energy"] for i in train_set.indices]
                print(train_labels)
                dataset_mean = torch.tensor(train_labels).mean(dim=0)
                dataset_std = torch.tensor(train_labels).std(dim=0)
                normalizer = Normalizer(dataset_mean, dataset_std)

                scaleNet_collect_fn = partial(scale_net_collate_fn, multiple_values_per_target=False)
                train_loader = DataLoader(
                    train_set,
                    batch_size=train_params["batch_size"],
                    shuffle=True,
                    collate_fn=scaleNet_collect_fn,
                    num_workers=train_params["num_workers"],
                    pin_memory=True,
                )
                valid_loader = DataLoader(
                    valid_set,
                    batch_size=train_params["test_batch_size"],
                    shuffle=False,
                    collate_fn=scaleNet_collect_fn,
                    num_workers=train_params["num_workers"],
                    pin_memory=True,
                )
                test_loader = DataLoader(
                    test_set,
                    batch_size=train_params["test_batch_size"],
                    shuffle=False,
                    collate_fn=scaleNet_collect_fn,
                    num_workers=train_params["num_workers"],
                    pin_memory=True,
                )

                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                structures, target_, outcar_id = scale_net_dataset[0][4:7]
                total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node = structures
                orig_atom_fea_len = total_atom_fea[0].shape[-1]
                nbr_fea_dist_len = total_nbr_dist_node[0].shape[-1]
                nbr_cat_value = 1

                model = ScaleNet(
                    elem_list=elem_list,
                    units=model_params["units"],
                    cutoff=model_params["cutoff"],
                    rbf_type="SphericalBessel",
                    nblocks=model_params["nblocks"],
                    use_smooth=True,
                    is_intensive=True,
                    field="node_feat",
                    orig_atom_fea_len=orig_atom_fea_len,
                    nbr_fea_dist_len=nbr_fea_dist_len,
                    nbr_cat_value=nbr_cat_value
                ).to(device)

                if pretrain_params["flag"]:
                    torch_path = pretrain_params["path"]
                    load_model(model, torch_path, device=device)
                    logger.info("Testing with Pretrained model")
                    predictions, labels = self.evaluate(device, model, test_loader, normalizer)
                    test_mae = mean_absolute_error(labels, predictions)
                    logger.info(f"Pretrained Test MAE {test_mae:.4f}")

                loss_fn = nn.SmoothL1Loss(reduction="mean")
                opt = optim.AdamW(
                    model.parameters(),
                    lr=train_params["lr"],
                    weight_decay=train_params["weight_decay"],
                    betas=(0.9, 0.999),
                    eps=1e-7,
                    amsgrad=True,
                )
                scheduler = LinearWarmupExponentialDecay(
                    opt,
                    train_params["warmup_steps"],
                    train_params["decay_steps"],
                    train_params["decay_rate"],
                    train_params["staircase"]
                )

                best_mae = 1e9
                no_improvement = 0
                ema_model = copy.deepcopy(model)
                for p in ema_model.parameters():
                    p.requires_grad_(False)
                best_model = copy.deepcopy(ema_model)
                train_loss_list = []
                valid_mae_list = []

                if train_params["mode"]:
                    logger.info("Training")
                    scaler = torch.cuda.amp.GradScaler()
                    for i in range(train_params["epochs"]):
                        train_loss = self.train(device, model, opt, loss_fn, train_params["grad_clip"], train_loader, scaler, normalizer)
                        train_loss_list.append(float(train_loss))
                        self.ema(ema_model, model, train_params["ema_decay"])
                        if i % train_params["interval"] == 0:
                            predictions, labels = self.evaluate(device, ema_model, valid_loader, normalizer)
                            valid_mae = mean_absolute_error(predictions, labels)
                            valid_mae_list.append(float(valid_mae))
                            logger.info(f"Epoch {i} | Train Loss {train_loss:.4f} | Val MAE {valid_mae:.4f}")
                            if valid_mae > best_mae:
                                no_improvement += 1
                                if no_improvement == train_params["early_stopping"]:
                                    logger.info("Early stop.")
                                    break
                            else:
                                no_improvement = 0
                                best_mae = valid_mae
                                best_model = copy.deepcopy(ema_model)
                                save_model(best_model, os.path.join(fold_save_dir, "best_model.pt"))
                        else:
                            logger.info(f"Epoch {i} | Train Loss {train_loss:.4f}")
                        scheduler.step()
                    save_model(ema_model, os.path.join(fold_save_dir, "last_model.pt"))
                    plot_model_metric(train_loss=train_loss_list, val_mae=valid_mae_list, save_dir=fold_save_dir, task=model_params["targets"][0])
                
                logger.info("Testing")
                predictions, labels = self.evaluate(device, best_model, test_loader, normalizer)
                test_mae = mean_absolute_error(labels, predictions)
                test_rmse = np.sqrt(mean_squared_error(labels, predictions))
                test_r2_score = r2_score(labels, predictions)
                plot_comparison_pic(labels.flatten(), predictions.flatten(), save_dir=fold_save_dir)
                logger.info(f"Fold {fold+1} Test MAE {test_mae:.4f}")
                logger.info(f"Fold {fold+1} Test RMSE {test_rmse:.4f}")
                logger.info(f"Fold {fold+1} Test R2_SCORE {test_r2_score:.4f}")
                fold_metrics.append({
                    'mae': test_mae,
                    'rmse': test_rmse,
                    'r2': test_r2_score
                })

            # Log overall cross-validation results
            logger.info("Cross-validation results:")
            mae_values = [m['mae'] for m in fold_metrics]
            rmse_values = [m['rmse'] for m in fold_metrics]
            r2_values = [m['r2'] for m in fold_metrics]
            logger.info(f"Average MAE: {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}")
            logger.info(f"Average RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")
            logger.info(f"Average R2: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")

            # Save cross-validation results
            cv_results = pd.DataFrame(fold_metrics)
            cv_results.to_csv(os.path.join(model_save_dir, "cv_results.csv"), index=False)
            cv_results.describe().to_csv(os.path.join(model_save_dir, "cv_summary.csv"))

        else:
            # Original single train/valid/test split code
            dataset_mean = torch.tensor(label_list).mean(dim=0)
            dataset_std = torch.tensor(label_list).std(dim=0)
            logger.info(f"The mean value of dataset: {dataset_mean}, and the standard value of dataset: {dataset_std}")
            pd.DataFrame({"mean": [float(dataset_mean)], "std": [float(dataset_std)]}).to_csv(os.path.join(model_save_dir, "data_distribution.csv"), index=False)
            normalizer = Normalizer(dataset_mean, dataset_std)
            
            train_data, valid_data, test_data = Trainer.split_dataset(
                scale_net_dataset,
                num_train=train_params["num_train"],
                num_valid=train_params["num_valid"],
                shuffle=True,
                random_state=seed,
            )
            logger.info(f"Size of Training Set: {len(train_data)}")
            logger.info(f"Size of Validation Set: {len(valid_data)}")
            logger.info(f"Size of Test Set: {len(test_data)}")

            scaleNet_collect_fn = partial(scale_net_collate_fn, multiple_values_per_target=False)
            train_loader = DataLoader(
                train_data,
                batch_size=train_params["batch_size"],
                shuffle=True,
                collate_fn=scaleNet_collect_fn,
                num_workers=train_params["num_workers"],
                pin_memory=True,
            )
            valid_loader = DataLoader(
                valid_data,
                batch_size=train_params["test_batch_size"],
                shuffle=False,
                collate_fn=scaleNet_collect_fn,
                num_workers=train_params["num_workers"],
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_data,
                batch_size=train_params["test_batch_size"],
                shuffle=False,
                collate_fn=scaleNet_collect_fn,
                num_workers=train_params["num_workers"],
                pin_memory=True,
            )

            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            structures, target_, outcar_id = scale_net_dataset[0][4:7]
            total_atom_fea, total_node_list, total_nbr_adj_node, total_nbr_dist_node, total_nbr_bond_cat, total_self_fea_idx, total_nbr_index_node = structures
            orig_atom_fea_len = total_atom_fea[0].shape[-1]
            nbr_fea_dist_len = total_nbr_dist_node[0].shape[-1]
            nbr_cat_value = 1

            model = ScaleNet(
                elem_list=elem_list,
                units=model_params["units"],
                cutoff=model_params["cutoff"],
                rbf_type="SphericalBessel",
                nblocks=model_params["nblocks"],
                use_smooth=True,
                is_intensive=True,
                field="node_feat",
                orig_atom_fea_len=orig_atom_fea_len,
                nbr_fea_dist_len=nbr_fea_dist_len,
                nbr_cat_value=nbr_cat_value
            ).to(device)

            if pretrain_params["flag"]:
                torch_path = pretrain_params["path"]
                load_model(model, torch_path, device=device)
                logger.info("Testing with Pretrained model")
                predictions, labels = self.evaluate(device, model, test_loader, normalizer)
                test_mae = mean_absolute_error(labels, predictions)
                logger.info(f"Pretrained Test MAE {test_mae:.4f}")

            loss_fn = nn.SmoothL1Loss(reduction="mean")
            opt = optim.AdamW(
                model.parameters(),
                lr=train_params["lr"],
                weight_decay=train_params["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-7,
                amsgrad=True,
            )
            scheduler = LinearWarmupExponentialDecay(
                opt,
                train_params["warmup_steps"],
                train_params["decay_steps"],
                train_params["decay_rate"],
                train_params["staircase"]
            )

            best_mae = 1e9
            no_improvement = 0
            ema_model = copy.deepcopy(model)
            for p in ema_model.parameters():
                p.requires_grad_(False)
            best_model = copy.deepcopy(ema_model)
            train_loss_list = []
            valid_mae_list = []

            if train_params["mode"]:
                logger.info("Training")
                scaler = torch.cuda.amp.GradScaler()
                for i in range(train_params["epochs"]):
                    train_loss = self.train(device, model, opt, loss_fn, train_params["grad_clip"], train_loader, scaler, normalizer)
                    train_loss_list.append(float(train_loss))
                    self.ema(ema_model, model, train_params["ema_decay"])
                    if i % train_params["interval"] == 0:
                        predictions, labels = self.evaluate(device, ema_model, valid_loader, normalizer)
                        valid_mae = mean_absolute_error(predictions, labels)
                        valid_mae_list.append(float(valid_mae))
                        logger.info(f"Epoch {i} | Train Loss {train_loss:.4f} | Val MAE {valid_mae:.4f}")
                        if valid_mae > best_mae:
                            no_improvement += 1
                            if no_improvement == train_params["early_stopping"]:
                                logger.info("Early stop.")
                                break
                        else:
                            no_improvement = 0
                            best_mae = valid_mae
                            best_model = copy.deepcopy(ema_model)
                            save_model(best_model, os.path.join(model_save_dir, "best_model.pt"))
                    else:
                        logger.info(f"Epoch {i} | Train Loss {train_loss:.4f}")
                    scheduler.step()
                save_model(ema_model, os.path.join(model_save_dir, "last_model.pt"))
                plot_model_metric(train_loss=train_loss_list, val_mae=valid_mae_list, save_dir=model_save_dir, task=model_params["targets"][0])
            
            logger.info("Testing")
            predictions, labels = self.evaluate(device, best_model, test_loader, normalizer)
            test_mae = mean_absolute_error(labels, predictions)
            test_rmse = np.sqrt(mean_squared_error(labels, predictions))
            test_r2_score = r2_score(labels, predictions)
            plot_comparison_pic(labels.flatten(), predictions.flatten(), save_dir=model_save_dir)
            logger.info("Test MAE {:.4f}".format(test_mae))
            logger.info("Test RMSE {:.4f}".format(test_rmse))
            logger.info("Test R2_SCORE {:.4f}".format(test_r2_score))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.main("./scaleNet.yaml")
