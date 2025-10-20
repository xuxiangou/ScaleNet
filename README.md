# ScaleNet: A framework predict nanoparticles with different size

<img src=".\paper_fig\Figure2.jpg" style="zoom:8%;" />

## Installation

This project required Pytorch, Pymatgen, Ase, DGL
Detailed version can see requirements.txt

## Train

For training and test, you need a dataset consist of structure files (cif format) and related properties data (id_prop.csv) in Google Cloude. Then, you can run training by following command:
```
python trainer.py
```

## Contents

|  Folder  |                         Description                         |
|          :------:          | :---------------------------------------------------------: |
|  `graph_theory_surface/`   |  Modified surgraph model (For larger nanoparticles feature extraction)  |
|  `local_gcn/`              |  Local extractor NN                                                     |
|  `best/`                   |  Best training model weights                                            |
|  `utils/`                  |  Tool method for model training and evaluation                          |
|  `other folder/`           |  global extractor NN methods                                            |


---

## Data

Training data: https://drive.google.com/file/d/1zlgWfOiJu-U0-E2BQkcJN2MVIm_YuBpw/view?usp=sharing
OOD test data: https://drive.google.com/file/d/1zlgWfOiJu-U0-E2BQkcJN2MVIm_YuBpw/view?usp=drive_link 

## How to change training parameters

You can modify the scaleNet.yaml to build your specific needs.

