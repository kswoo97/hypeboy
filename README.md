# HypeBoy: Generative Self-Supervised Representation Learning on Hypergraphs

### Authors: Sunwoo Kim, Shinhwan Kang, Fanchen Bu, Soo Yong Lee, Jaemin Yoo, and Kijung Shin.
### Paper: https://openreview.net/forum?id=DZUzOKE6og

Published as a conference paper at ICLR 2024

## Dataset Description

We provide 11 hypergraph benchmark datasets that are used in our work. 
Dataset statistics are as follows:
| Dataset | # of nodes | # of hyperedges | # of features | # of classes |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Cora | 1,434 | 1,579 | 1,433 | 7 |
| Citeseer | 1,458 | 1,079 | 3,703 | 6 |
| Pubmed | 3,840 | 7,963 | 500 | 3 |
| Cora-CA | 2,388 | 1,072 | 1,433 | 7 |
| DBLP-P | 41,302 | 22,263 | 1,425 | 6 |
| DBLP-A | 2,591 | 2,690 | 334 | 4 |
| AMiner | 20,201 | 8,052 | 500 | 12 |
| IMDB | 3,939 | 2,015 | 3066 | 3 |
| ModelNet-40 | 12,311 | 12,311 | 100 | 40 |
| News | 16,242 | 100 | 4 | 4 |
| House | 1,290 | 341 | 100 | 2 |

(Due to the file size of datasets) Datasets can be found at https://www.dropbox.com/scl/fi/pes0jgk0syrxosk71n1ua/datasets.zip?rlkey=h1r7j745z1a73qlrd6pv9ypcd&dl=0.

Details of each dataset file are provided at README.txt in the provided datasets.zip of the above link.

## Code Description

### Overview
We provide code for reproducing HypeBoy's performance at:
- Fine-tuning of node classification,
- Linear evaluation of node classification,
- Linear evaluation of hyperedge prediction.

### Pre-requisite
The overall hierarchy should be set as follows:
```
/datasets
  |
  /cora_cite
    |
    X.pt
    H.pt
    Y.pt
    edge_bucket.pickle
    data_split_0.01.pickle
  /citeseer_cite
    |...
  ...
main.py
src.py
HNNs.py
```

### Run code
The crucial command is as follows:
```
python3 main.py -data "str: Data name" -task "str: Downstream task" -epoch "int: SSL epoch" -p_x "float: $p_{x}$" -p_e "float: $p_{e}$" -device "str: Cuda device"
```

Example:
```
python3 main.py -data cora_cite -task finetuning -epoch 200 -p_x 0.4 -p_e 0.8 -device cuda:0
```
Details of each component are as follows:

**-data (String):**
Name of datasets you want to use. It should be given one of
- cora_cite: Cora dataset
- citeseer_cite: Citeseer dataset
- pubmed_cite: Pubmed dataset
- cora_ca: Cora-Coauthorship dataset
- dblp_coauth: DBLP-P dataset
- dblp_copub: DBLP-A dataset
- aminer: AMiner dataset
- imdb: IMDB dataset
- modelnet_40: ModeltNet 40 dataset
- news: News dataset
- house: House dataset

E.g., -data cora_cite

**-task (String):**
The task you want to evaluate. It should be given one of
- finetuning: Fintuning protocol of node classification
- linear_node: Linear evaluation protocol of node classification
- linear_edge: Linear evaluation protocol of hyperedge prediction

E.g., -task finetuning

**-epoch (Integer):**
Number of epochs you want to run HypeBoy. It should be given as an integer.

E.g., -epoch 200

**-p_x (Float):**
Node feature drop probability $p_{x}$. Note that $p_{x} \in [0, 1]$ should hold.

E.g., -p_x 0.4

**-p_x (Float):**
Hyperedge drop probability $p_{e}$. Note that $p_{e} \in [0, 1]$ should hold.

E.g., -p_x 0.5

**-device (String):**
Cuda device machine name.

E.g., -device cuda:0
