"""
gui.py — ScaleNet 吸附能预测可视化界面
亮色主题 · 中文界面 · 支持带标签对比 / 纯文件夹批量预测两种模式
"""

import os, sys, traceback
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch, dgl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from joblib import Parallel, delayed
from tqdm import tqdm
from pymatgen.io.cif import CifParser
from ruamel.yaml import YAML
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QProgressBar,
    QTableWidget, QTableWidgetItem, QSplitter, QFrame, QTabWidget,
    QDoubleSpinBox, QSpinBox, QGroupBox, QGridLayout,
    QHeaderView, QTextEdit, QSizePolicy, QRadioButton,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from scaleNet import ScaleNet
from train_utils.utils import load_model
from ScaleNetDataset import ScaleNetDataset
from ext.pymatgen import Structure2Graph, get_element_list


# ═══════════════════════════════════════════════════════════════════
# 调色板（亮色·学术风）
# ═══════════════════════════════════════════════════════════════════
BG     = "#F4F6FA"
PANEL  = "#FFFFFF"
BORDER = "#DDE3ED"
BLUE   = "#2563EB"
BLUE_L = "#EFF6FF"
TEAL   = "#0891B2"
GREEN  = "#16A34A"
RED    = "#DC2626"
TEXT   = "#1E293B"
MUTED  = "#64748B"
LIGHT  = "#94A3B8"

QSS = f"""
* {{ font-family: "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif; }}
QMainWindow, QWidget {{ background:{BG}; color:{TEXT}; }}

QFrame#card {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:10px;
}}
QLabel#h1  {{ font-size:20px; font-weight:700; color:{TEXT}; }}
QLabel#cap {{ font-size:11px; color:{MUTED}; }}
QLabel#mv  {{ font-size:26px; font-weight:800; color:{BLUE}; }}
QLabel#ml  {{ font-size:10px; color:{MUTED}; letter-spacing:0.8px; }}
QLabel#badge {{
    font-size:10px; color:{BLUE}; background:{BLUE_L};
    border-radius:4px; padding:2px 8px;
}}

QPushButton#primary {{
    background:{BLUE}; color:white; border:none;
    border-radius:7px; padding:9px 22px; font-size:13px; font-weight:600;
}}
QPushButton#primary:hover   {{ background:#1D4ED8; }}
QPushButton#primary:pressed {{ background:#1E40AF; }}
QPushButton#primary:disabled {{ background:{BORDER}; color:{LIGHT}; }}
QPushButton#outline {{
    background:transparent; color:{BLUE}; border:1.5px solid {BLUE};
    border-radius:7px; padding:7px 16px; font-size:12px; font-weight:500;
}}
QPushButton#outline:hover {{ background:{BLUE_L}; }}
QPushButton#ghost {{
    background:transparent; color:{MUTED}; border:1px solid {BORDER};
    border-radius:6px; padding:5px 10px; font-size:11px;
}}
QPushButton#ghost:hover {{ background:{BG}; color:{TEXT}; }}

QLineEdit, QDoubleSpinBox, QSpinBox {{
    background:{PANEL}; border:1.5px solid {BORDER};
    border-radius:6px; padding:6px 10px; color:{TEXT}; font-size:12px;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color:{BLUE}; background:#FAFCFF;
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button,       QSpinBox::down-button {{
    border:none; background:{BG}; width:16px;
}}

QProgressBar {{
    background:{BORDER}; border:none; border-radius:5px;
    height:8px; text-align:center; color:transparent;
}}
QProgressBar::chunk {{ background:{BLUE}; border-radius:5px; }}

QTableWidget {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:8px;
    gridline-color:{BG}; color:{TEXT}; font-size:12px;
    selection-background-color:{BLUE_L}; selection-color:{TEXT};
}}
QTableWidget::item {{ padding:4px 8px; }}
QTableWidget::item:alternate {{ background:#FAFBFD; }}
QHeaderView::section {{
    background:{BG}; color:{MUTED}; border:none;
    border-bottom:1px solid {BORDER}; padding:7px 10px;
    font-size:10px; font-weight:600; letter-spacing:0.8px;
}}

QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:8px; background:{PANEL}; }}
QTabBar::tab {{
    background:{BG}; color:{MUTED}; border:1px solid {BORDER};
    border-bottom:none; border-radius:6px 6px 0 0;
    padding:7px 18px; font-size:12px; margin-right:3px;
}}
QTabBar::tab:selected {{ background:{PANEL}; color:{BLUE}; font-weight:600; }}

QTextEdit {{
    background:#F8FAFC; border:1px solid {BORDER}; border-radius:7px;
    color:{MUTED}; font-size:11px;
    font-family:"Consolas","Courier New",monospace; padding:6px 10px;
}}

QGroupBox {{
    border:1.5px solid {BORDER}; border-radius:8px; margin-top:14px;
    font-size:12px; font-weight:600; color:{MUTED};
}}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 4px; }}

QRadioButton {{ font-size:12px; color:{TEXT}; spacing:6px; }}
QRadioButton::indicator {{
    width:14px; height:14px; border:1.5px solid {BORDER};
    border-radius:7px; background:{PANEL};
}}
QRadioButton::indicator:checked {{ background:{BLUE}; border-color:{BLUE}; }}

QScrollBar:vertical {{ background:{BG}; width:8px; border-radius:4px; }}
QScrollBar::handle:vertical {{
    background:{BORDER}; border-radius:4px; min-height:24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""

# matplotlib 全局亮色样式
plt.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": BG,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "grid.color": BORDER, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "font.size": 9,
})


# ═══════════════════════════════════════════════════════════════════
# 数据工具（继承原始逻辑）
# ═══════════════════════════════════════════════════════════════════
class Normalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(float(mean))
        self.std  = torch.tensor(float(std))
    def norm(self, x):   return (x - self.mean) / self.std
    def denorm(self, x): return x * self.std + self.mean


def load_data_with_labels(data_dir):
    label_dict = pd.read_csv(
        os.path.join(data_dir, "id_prop.csv"), header=None, index_col=0
    ).to_dict()[1]
    def _r(f, l):
        s = CifParser(os.path.join(data_dir, str(f)+".cif"),
                      frac_tolerance=0).parse_structures()[0]
        return f, s, l
    res = Parallel(n_jobs=8)(delayed(_r)(f,l) for f,l in tqdm(label_dict.items()))
    ids, strucs, labs = zip(*res)
    return list(strucs), list(labs), list(ids)


def load_data_folder(cif_dir):
    files = sorted(Path(cif_dir).glob("*.cif"))
    if not files:
        raise FileNotFoundError(f"目录中未找到 .cif 文件：{cif_dir}")
    def _r(p):
        s = CifParser(str(p), frac_tolerance=0).parse_structures()[0]
        return p.stem, s
    res = Parallel(n_jobs=8)(delayed(_r)(p) for p in tqdm(files))
    ids, strucs = zip(*res)
    return list(strucs), list(ids)


def scale_net_collate_fn(batches, multiple_values_per_target=False):
    g   = dgl.batch([b[0] for b in batches])
    lat = torch.stack([b[1][0] for b in batches])
    sa  = torch.stack([b[2] for b in batches])
    lbs = torch.stack([list(b[3].values())[0] for b in batches])

    def _cgcnn(batch):
        ba,bd,bn,bb,bsi,bni,ai,ci,bt,oc,strc = [],[],[],[],[],[],[],[],[],[],[]
        base = 0
        for i, ((afea,nlist,nadj,ndist,nbond,sidx,nidx), tgt, oid, st) in enumerate(batch):
            total_n = sum(len(nlist[j]) for j in range(len(nlist)))
            for a in range(len(nlist)):
                n_i = len(nlist[a])
                ba.append(afea[a]); bd.append(ndist[a])
                bn.append(nadj[a]);  bb.append(nbond[a])
                bsi.extend([sidx[a]+base]); bni.extend([nidx[a]+base])
                ai.extend([a+base]*n_i); base += n_i
            ci.extend([i]*total_n)
            bt.append(tgt); oc.append(oid); strc.append(st)
        return ((torch.cat(ba), torch.cat(bd), torch.cat(bn), torch.cat(bb),
                 torch.cat(bsi), torch.cat(bni),
                 torch.LongTensor(ai), torch.LongTensor(ci)),
                torch.stack(bt), oc, strc)

    return (g, lat, sa, lbs,
            *_cgcnn([b[4:8] for b in batches]),
            *_cgcnn([b[8:]  for b in batches]))


@torch.no_grad()
def evaluate(device, model, loader, normalizer):
    model.eval()
    preds, labs, strucs = [], [], []
    for g, lat, sa, labels, inp, _, _, structures, inp2, _, _, _ in loader:
        g.edata["lattice"] = torch.repeat_interleave(lat, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (
            g.edata["pbc_offset"].unsqueeze(-1) * g.edata["lattice"]
        ).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(-1) *
            torch.repeat_interleave(lat, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        out = model(g.to(device), sa.to(device),
                    (x.to(device) for x in inp),
                    (x.to(device) for x in inp2))
        labs.extend(labels)
        preds.extend(normalizer.denorm(out.view(-1).cpu()).numpy())
        strucs.extend(structures)
    return preds, labs, strucs


# ═══════════════════════════════════════════════════════════════════
# Worker 线程
# ═══════════════════════════════════════════════════════════════════
class PredictWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg

    def run(self):
        try:
            c = self.cfg
            mode = c["mode"]

            self.progress.emit(10, "加载结构文件…")
            if mode == "eval":
                structures, labels, file_ids = load_data_with_labels(c["data_dir"])
            else:
                structures, file_ids = load_data_folder(c["data_dir"])
                labels = [0.0] * len(structures)

            # 分离零吸附结构（仅评估模式）
            zero_s = zero_l = None
            if mode == "eval":
                for i, s in enumerate(structures):
                    if s.composition.get("O", 0) == 0:
                        zero_s, zero_l = s, labels[i]
                        structures.pop(i); labels.pop(i); file_ids.pop(i)
                        break

            self.progress.emit(30, "构建图数据集…")
            elem_list = get_element_list(structures)
            converter = Structure2Graph(element_types=elem_list, cutoff=c["cutoff"])
            dataset   = ScaleNetDataset(
                threebody_cutoff=c["cutoff"], structures=structures,
                graph_converter=converter, labels=labels,
                file_id_list=file_ids, include_line_graph=False, return_struc=True,
            )
            normalizer = Normalizer(c["mean"], c["std"])
            loader = DataLoader(
                dataset, batch_size=c["batch_size"], shuffle=False,
                collate_fn=partial(scale_net_collate_fn, multiple_values_per_target=False),
                num_workers=c["num_workers"], pin_memory=False,
            )

            self.progress.emit(55, "加载模型权重…")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gi, _, _, _ = dataset[0][4:8]
            afea, _, _, ndist, _, _, _ = gi
            model = ScaleNet(
                elem_list=elem_list, units=c["units"], cutoff=c["cutoff"],
                rbf_type="SphericalBessel", nblocks=c["nblocks"],
                use_smooth=True, is_intensive=True, field="node_feat",
                orig_atom_fea_len=afea[0].shape[-1],
                nbr_fea_dist_len=ndist[0].shape[-1], nbr_cat_value=1,
            ).to(device)
            load_model(model, c["model_path"], device=device)

            self.progress.emit(75, "模型推理中…")
            preds, labs, strucs = evaluate(device, model, loader, normalizer)

            if mode == "eval" and zero_s is not None:
                strucs.append(zero_s); labs.append(zero_l); preds.append(zero_l)

            names = [
                f"Cu{int(s.composition.get('Cu',0))}"
                f"(OH){int(s.composition.get('O',0))}"
                for s in strucs
            ]
            result = {
                "mode": mode, "names": names,
                "predictions": preds,
                "labels": [float(torch.tensor(l)) for l in labs],
                "device": str(device),
                "mae": None, "r2": None,
            }
            if mode == "eval":
                n = len(preds) - (1 if zero_s else 0)
                result["mae"] = mean_absolute_error(labs[:n], preds[:n])
                result["r2"]  = r2_score(labs[:n], preds[:n])

            self.progress.emit(100, "预测完成 ✓")
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════
# Matplotlib 画布
# ═══════════════════════════════════════════════════════════════════
class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 4), facecolor=PANEL, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._empty()

    def _empty(self):
        self.ax.clear(); self.ax.set_facecolor(BG)
        self.ax.text(0.5, 0.5, "暂无数据，请先运行预测",
                     transform=self.ax.transAxes,
                     ha="center", va="center", color=LIGHT, fontsize=12)
        self.ax.axis("off"); self.draw()

    def _reset(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG)
        for sp in self.ax.spines.values():
            sp.set_color(BORDER)
        self.ax.tick_params(colors=MUTED, labelsize=8)

    def plot_parity(self, labels, preds, mae, r2):
        self._reset()
        lo = min(min(labels), min(preds)) - 0.15
        hi = max(max(labels), max(preds)) + 0.15
        self.ax.plot([lo,hi],[lo,hi],"--",color=LIGHT,lw=1.2,zorder=1)
        err = np.abs(np.array(labels) - np.array(preds))
        sc  = self.ax.scatter(labels, preds, c=err, cmap="RdYlGn_r",
                              s=36, alpha=0.85, edgecolors="white",
                              linewidths=0.4, zorder=3)
        cb = self.fig.colorbar(sc, ax=self.ax, fraction=0.03, pad=0.02)
        cb.set_label("|误差| (eV)", color=MUTED, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        self.ax.set_xlabel("DFT 标签 (eV)"); self.ax.set_ylabel("ML 预测 (eV)")
        self.ax.set_title(f"预测 vs DFT 奇偶图    MAE={mae:.4f} eV    R²={r2:.4f}",
                          color=TEXT, fontsize=9.5, pad=10)
        self.ax.grid(True); self.draw()

    def plot_bar(self, names, preds, labels, show_labels):
        self._reset()
        x = np.arange(len(names)); w = 0.38
        if show_labels:
            self.ax.bar(x-w/2, labels, w, color=GREEN, alpha=0.75, label="DFT", zorder=3)
        self.ax.bar(x+(w/2 if show_labels else 0), preds, w,
                    color=BLUE, alpha=0.80, label="ML 预测", zorder=3)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        self.ax.set_ylabel("吸附能 (eV)")
        self.ax.set_title("各结构吸附能预测结果", color=TEXT, fontsize=9.5, pad=10)
        if show_labels:
            self.ax.legend(framealpha=0.9, fontsize=8)
        self.ax.grid(axis="y"); self.fig.tight_layout(); self.draw()

    def plot_hist(self, preds, labels):
        self._reset()
        errors = np.array(preds) - np.array(labels)
        self.ax.hist(errors, bins=20, color=BLUE, alpha=0.72,
                     edgecolor="white", linewidth=0.5)
        self.ax.axvline(0, color=RED, lw=1.2, linestyle="--", label="零误差线")
        self.ax.set_xlabel("预测误差 (eV)"); self.ax.set_ylabel("频次")
        self.ax.set_title("误差分布直方图", color=TEXT, fontsize=9.5, pad=10)
        self.ax.legend(fontsize=8); self.ax.grid(axis="y"); self.draw()


# ═══════════════════════════════════════════════════════════════════
# 左侧参数面板
# ═══════════════════════════════════════════════════════════════════
class ParamPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(305)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 6, 0); lay.setSpacing(10)

        # 模式
        mb = QGroupBox("预测模式"); ml = QVBoxLayout(mb)
        self.rb_eval   = QRadioButton("评估模式（含 DFT 标签对比）")
        self.rb_folder = QRadioButton("批量预测（纯 CIF 文件夹）")
        self.rb_eval.setChecked(True)
        ml.addWidget(self.rb_eval); ml.addWidget(self.rb_folder)
        lay.addWidget(mb)

        # 路径
        pb = QGroupBox("路径设置"); pl = QGridLayout(pb); pl.setVerticalSpacing(6)
        pl.addWidget(self._cap("数据目录"), 0, 0, 1, 2)
        self.data_dir = QLineEdit("./pourbaix_data/Cu_7A")
        self.data_dir.setPlaceholderText("含 .cif 的文件夹")
        b1 = self._btn("…"); b1.clicked.connect(
            lambda: self._browse_dir(self.data_dir))
        pl.addWidget(self.data_dir, 1, 0); pl.addWidget(b1, 1, 1)
        pl.addWidget(self._cap("模型权重 (.pt)"), 2, 0, 1, 2)
        self.model_path = QLineEdit("./model/best/best_model_1.pt")
        b2 = self._btn("…"); b2.clicked.connect(
            lambda: self._browse_file(self.model_path, "模型文件 (*.pt *.pth)"))
        pl.addWidget(self.model_path, 3, 0); pl.addWidget(b2, 3, 1)
        lay.addWidget(pb)

        # 归一化
        nb = QGroupBox("归一化参数"); nl = QGridLayout(nb); nl.setVerticalSpacing(6)
        nl.addWidget(self._cap("均值 mean"), 0, 0)
        self.mean_sp = QDoubleSpinBox()
        self.mean_sp.setRange(-1e6,1e6); self.mean_sp.setDecimals(6)
        self.mean_sp.setSingleStep(0.01); self.mean_sp.setValue(-3.225145)
        nl.addWidget(self.mean_sp, 0, 1)
        nl.addWidget(self._cap("标准差 std"), 1, 0)
        self.std_sp = QDoubleSpinBox()
        self.std_sp.setRange(1e-9,1e6); self.std_sp.setDecimals(6)
        self.std_sp.setSingleStep(0.01); self.std_sp.setValue(2.925980)
        nl.addWidget(self.std_sp, 1, 1)
        lay.addWidget(nb)

        # 模型结构
        ab = QGroupBox("模型结构"); al = QGridLayout(ab); al.setVerticalSpacing(6)
        al.addWidget(self._cap("截断半径 cutoff (Å)"), 0, 0)
        self.cutoff_sp = QDoubleSpinBox()
        self.cutoff_sp.setRange(2.0,20.0); self.cutoff_sp.setSingleStep(0.5)
        self.cutoff_sp.setValue(5.0); self.cutoff_sp.setSuffix(" Å")
        al.addWidget(self.cutoff_sp, 0, 1)
        al.addWidget(self._cap("隐藏维度 units"), 1, 0)
        self.units_sp = QSpinBox()
        self.units_sp.setRange(8,512); self.units_sp.setSingleStep(8)
        self.units_sp.setValue(64)
        al.addWidget(self.units_sp, 1, 1)
        al.addWidget(self._cap("网络层数 nblocks"), 2, 0)
        self.nblocks_sp = QSpinBox()
        self.nblocks_sp.setRange(1,12); self.nblocks_sp.setValue(3)
        al.addWidget(self.nblocks_sp, 2, 1)
        lay.addWidget(ab)

        # 推理
        ib = QGroupBox("推理参数"); il = QGridLayout(ib); il.setVerticalSpacing(6)
        il.addWidget(self._cap("批大小 batch_size"), 0, 0)
        self.batch_sp = QSpinBox()
        self.batch_sp.setRange(1,256); self.batch_sp.setValue(2)
        il.addWidget(self.batch_sp, 0, 1)
        il.addWidget(self._cap("数据加载线程"), 1, 0)
        self.workers_sp = QSpinBox()
        self.workers_sp.setRange(0,32); self.workers_sp.setValue(2)
        il.addWidget(self.workers_sp, 1, 1)
        lay.addWidget(ib)

        lay.addStretch()

    def _cap(self, t):
        l = QLabel(t); l.setObjectName("cap"); return l

    def _btn(self, t):
        b = QPushButton(t); b.setObjectName("ghost"); b.setFixedWidth(28); return b

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        if d: edit.setText(d)

    def _browse_file(self, edit, filt):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", edit.text(), filt)
        if f: edit.setText(f)

    def collect(self):
        return {
            "mode":       "eval" if self.rb_eval.isChecked() else "folder",
            "data_dir":   self.data_dir.text().strip(),
            "model_path": self.model_path.text().strip(),
            "mean":       self.mean_sp.value(),
            "std":        self.std_sp.value(),
            "cutoff":     self.cutoff_sp.value(),
            "units":      self.units_sp.value(),
            "nblocks":    self.nblocks_sp.value(),
            "batch_size": self.batch_sp.value(),
            "num_workers":self.workers_sp.value(),
        }


# ═══════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScaleNet · 吸附能预测系统")
        self.setMinimumSize(1160, 750)
        self.setStyleSheet(QSS)
        self.results = None; self.worker = None
        self._build()

    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        rl = QVBoxLayout(root)
        rl.setContentsMargins(16, 12, 16, 12); rl.setSpacing(10)

        # ── 标题栏 ──────────────────────────────
        hdr = QHBoxLayout()
        col = QVBoxLayout(); col.setSpacing(1)
        t = QLabel("ScaleNet"); t.setObjectName("h1"); col.addWidget(t)
        s = QLabel("图神经网络吸附能预测系统")
        s.setObjectName("cap"); col.addWidget(s)
        hdr.addLayout(col); hdr.addStretch()
        self.dev_lbl = QLabel("设备：—")
        self.dev_lbl.setObjectName("badge"); hdr.addWidget(self.dev_lbl)
        rl.addLayout(hdr)

        # ── 分割线 + 主体 ───────────────────────
        body = QHBoxLayout(); body.setSpacing(14)
        self.params = ParamPanel()
        body.addWidget(self.params)

        right = QVBoxLayout(); right.setSpacing(10)

        # 运行栏
        run_card = QFrame(); run_card.setObjectName("card")
        rc = QHBoxLayout(run_card); rc.setContentsMargins(14,10,14,10)
        self.run_btn = QPushButton("▶  开始预测")
        self.run_btn.setObjectName("primary")
        self.run_btn.setFixedWidth(128)
        self.run_btn.clicked.connect(self._run)
        rc.addWidget(self.run_btn)
        self.prog = QProgressBar(); self.prog.setValue(0)
        rc.addWidget(self.prog, stretch=1)
        self.status = QLabel("就绪")
        self.status.setObjectName("cap"); self.status.setFixedWidth(155)
        rc.addWidget(self.status)
        right.addWidget(run_card)

        # 指标卡片
        mr = QHBoxLayout(); mr.setSpacing(10)
        self.mae_card, self.mae_v = self._metric("平均绝对误差 MAE", "—", "eV")
        self.r2_card,  self.r2_v  = self._metric("决定系数 R²",     "—", "")
        self.n_card,   self.n_v   = self._metric("样本总数",         "—", "个")
        for c in (self.mae_card, self.r2_card, self.n_card):
            mr.addWidget(c)
        right.addLayout(mr)

        # Tab 面板
        self.tabs = QTabWidget()

        # Tab 1：结果表格
        t1 = QWidget(); l1 = QVBoxLayout(t1); l1.setContentsMargins(6,6,6,6)
        tr = QHBoxLayout(); tr.addWidget(QLabel("预测结果列表"))
        tr.addStretch()
        self.exp_btn = QPushButton("导出 CSV")
        self.exp_btn.setObjectName("outline")
        self.exp_btn.setEnabled(False)
        self.exp_btn.clicked.connect(self._export)
        tr.addWidget(self.exp_btn)
        l1.addLayout(tr)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["结构名称", "DFT 标签 (eV)", "ML 预测 (eV)", "误差 (eV)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1,2,3):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        l1.addWidget(self.table)
        self.tabs.addTab(t1, "📋  结果表格")

        # Tab 2：可视化
        t2 = QWidget(); l2 = QVBoxLayout(t2); l2.setContentsMargins(6,6,6,6)
        br = QHBoxLayout(); br.setSpacing(6)
        self.bp = QPushButton("奇偶图");  self.bp.setObjectName("outline")
        self.bb = QPushButton("柱状图");  self.bb.setObjectName("outline")
        self.bh = QPushButton("误差分布"); self.bh.setObjectName("outline")
        for b in (self.bp, self.bb, self.bh):
            b.setFixedHeight(30); br.addWidget(b)
        br.addStretch()
        self.bp.clicked.connect(lambda: self._plot("parity"))
        self.bb.clicked.connect(lambda: self._plot("bar"))
        self.bh.clicked.connect(lambda: self._plot("hist"))
        l2.addLayout(br)
        self.canvas = PlotCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        l2.addWidget(self.canvas)
        self.tabs.addTab(t2, "📊  可视化")

        right.addWidget(self.tabs, stretch=1)

        # 日志
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.log.setMaximumHeight(66)
        right.addWidget(self.log)

        body.addLayout(right, stretch=1)
        rl.addLayout(body)

    def _metric(self, label, val, unit):
        c = QFrame(); c.setObjectName("card")
        l = QVBoxLayout(c); l.setContentsMargins(16,12,16,12); l.setSpacing(2)
        lb = QLabel(label); lb.setObjectName("ml"); l.addWidget(lb)
        vl = QLabel(val);   vl.setObjectName("mv"); l.addWidget(vl)
        ul = QLabel(unit);  ul.setObjectName("cap"); l.addWidget(ul)
        return c, vl

    def _run(self):
        cfg = self.params.collect()
        if not os.path.isdir(cfg["data_dir"]):
            self._log(f"⚠ 数据目录不存在：{cfg['data_dir']}", err=True); return
        if not os.path.isfile(cfg["model_path"]):
            self._log(f"⚠ 模型文件不存在：{cfg['model_path']}", err=True); return
        self.run_btn.setEnabled(False); self.exp_btn.setEnabled(False)
        self.prog.setValue(0); self.table.setRowCount(0)
        self.canvas._empty()
        self._log(f"启动预测 · 模式：{'评估' if cfg['mode']=='eval' else '批量预测'}")
        self.worker = PredictWorker(cfg)
        self.worker.progress.connect(self._on_prog)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_prog(self, v, msg):
        self.prog.setValue(v); self.status.setText(msg); self._log(msg)

    def _on_done(self, res):
        self.results = res
        self.run_btn.setEnabled(True); self.exp_btn.setEnabled(True)
        mode = res["mode"]
        self.dev_lbl.setText(f"设备：{res['device']}")
        self.mae_v.setText(f"{res['mae']:.4f}" if mode=="eval" else "N/A")
        self.r2_v.setText(f"{res['r2']:.4f}"  if mode=="eval" else "N/A")
        self.n_v.setText(str(len(res["names"])))

        preds = res["predictions"]; labs = res["labels"]
        self.table.setRowCount(len(res["names"]))
        for i, (nm, lb, pr) in enumerate(zip(res["names"], labs, preds)):
            err = pr - lb
            row = [nm,
                   f"{lb:.4f}" if mode=="eval" else "—",
                   f"{pr:.4f}",
                   f"{err:+.4f}" if mode=="eval" else "—"]
            for j, txt in enumerate(row):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if j == 3 and mode == "eval":
                    item.setForeground(
                        QColor(RED) if abs(err) > 0.1 else QColor(GREEN))
                self.table.setItem(i, j, item)

        self._plot("parity" if mode=="eval" else "bar")
        self.tabs.setCurrentIndex(1)
        self._log("✅ 完成  " + (
            f"样本={len(res['names'])}  MAE={res['mae']:.4f}  R²={res['r2']:.4f}"
            if mode=="eval" else f"样本={len(res['names'])}"))

    def _on_err(self, tb):
        self.prog.setValue(0); self.status.setText("发生错误")
        self.run_btn.setEnabled(True); self._log(tb, err=True)

    def _plot(self, kind):
        if not self.results: return
        mode  = self.results["mode"]
        preds = self.results["predictions"]
        labs  = self.results["labels"]
        names = self.results["names"]
        if   kind == "parity" and mode == "eval":
            self.canvas.plot_parity(labs, preds, self.results["mae"], self.results["r2"])
        elif kind == "bar":
            self.canvas.plot_bar(names, preds, labs, show_labels=(mode=="eval"))
        elif kind == "hist" and mode == "eval":
            self.canvas.plot_hist(preds, labs)
        else:
            self.canvas.plot_bar(names, preds, labs, show_labels=False)

    def _export(self):
        if not self.results: return
        path, _ = QFileDialog.getSaveFileName(self,"保存 CSV","预测结果.csv","CSV (*.csv)")
        if path:
            mode = self.results["mode"]
            d = {"结构名称": self.results["names"],
                 "ML预测(eV)": self.results["predictions"]}
            if mode == "eval":
                d["DFT标签(eV)"] = self.results["labels"]
                d["误差(eV)"] = [p-l for p,l in zip(
                    self.results["predictions"], self.results["labels"])]
            pd.DataFrame(d).to_csv(path, index=False, encoding="utf-8-sig")
            self._log(f"📁 已导出 → {path}")

    def _log(self, msg, err=False):
        c = RED if err else MUTED
        self.log.append(f'<span style="color:{c};">{msg}</span>')


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("PingFang SC", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())