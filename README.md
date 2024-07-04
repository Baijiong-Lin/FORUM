# FORUM

[Feiyang Ye](https://feiyang-ye.github.io/), [Baijiong Lin](https://baijiong-lin.github.io/), Xiaofeng Cao, [Yu Zhang](https://yuzhanghk.github.io/), and Ivor Tsang. A First-Order Multi-Gradient Algorithm for Multi-Objective Bi-Level Optimization. In *European Conference on Artificial Intelligence*, 2024.



## Highlights

1. FORUM: a more effective and efficient solution for multi-objective bi-level optimization problems;

2. A more efficient implementation for [MOML](https://github.com/Baijiong-Lin/MOML) (but it is only applicable to multi-task learning). 
   
   

## Installation

1. Create a virtual environment
   
   ```shell
   conda create -n forum python=3.8
   conda activate forum
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Clone the repository
   
   ```shell
   git clone https://github.com/Baijiong-Lin/FORUM.git
   ```

3. Install `LibMTL`
   
   ```shell
   cd FORUM
   pip install -r requirements.txt
   pip install -e .
   ```
   
   

## Training

1. NYUv2 dataset: download the data from [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0), and then run the following command for training,
   
   ```shell
   cd examples/nyu
   python train_nyu.py --dataset_path /path/to/ --scheduler step --method FORUM --rho 0.1 --eta 0.1 --inner_step 5 ## FORUM
   python train_nyu.py --dataset_path /path/to/ --scheduler step --method MOML --weighting MGDA --eta 0.1 ## MOML
   ```

2. Office-31 and Office-Home datasets: download the data from [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/#datasets_code) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), and then run the following command for training,
   
   ```shell
   cd examples/office
   python train_office.py --dataset office-31 --dataset_path /path/to/ --multi_input --method FORUM --rho 0.1 --eta 0.1 --inner_step 5 ## FORUM, Office-31
   python train_office.py --dataset office-31 --dataset_path /path/to/ --multi_input --method MOML --weighting MGDA --eta 0.1 ## MOML, Office-31
   ```

3. QM9 dataset: the data will be downloaded automatically, thus directly running the following command for training,
   
   ```shell
   cd examples/qm9
   python train_qm9.py --dataset_path /path/to/ --method FORUM --rho 0.1 --inner_step 5 --eta 0.01 --lr 0.001 --weight_decay 0 ## FORUM
   python train_qm9.py --dataset_path /path/to/ --method MOML --weighting MGDA --eta 0.01 --lr 0.001 --weight_decay 0 ## MOML
   ```
   
   

Acknowledgement
---------------

This code is heavily based on [LibMTL](https://github.com/median-research-group/LibMTL).



## Citation

If you find this work/code useful for your research, please cite the following:

```latex
@inproceedings{ye2024forum,
  title={A First-Order Multi-Gradient Algorithm for Multi-Objective Bi-Level Optimization},
  author={Ye, Feiyang and Lin, Baijiong and Cao, Xiaofeng and Zhang, Yu and Tsang, Ivor},
  booktitle={European Conference on Artificial Intelligence},
  year={2024}
}

@article{lin2023libmtl,
  title={{LibMTL}: A {P}ython Library for Multi-Task Learning},
  author={Baijiong Lin and Yu Zhang},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={209},
  pages={1--7},
  year={2023}
}
```
