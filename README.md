# GNNExplainer

CZ4071 - Network Science

This repository contains the modified code for the paper `GNNExplainer: Generating Explanations for Graph Neural Networks` by [Rex Ying](https://cs.stanford.edu/people/rexy/), [Dylan Bourgeois](https://dtsbourg.me/), [Jiaxuan You](https://cs.stanford.edu/~jiaxuan/), [Marinka Zitnik](http://helikoid.si/cms/) & [Jure Leskovec](https://cs.stanford.edu/people/jure/), presented at [NeurIPS 2019](nips.cc).

[[Arxiv]](https://arxiv.org/abs/1903.03894)
```
@misc{ying2019gnnexplainer,
    title={GNNExplainer: Generating Explanations for Graph Neural Networks},
    author={Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec},
    year={2019},
    eprint={1903.03894},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
The code involves expirements to run benchmarks on GNN explainability on synthetic graphs generated using [Barabási–Albert model](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model)

## Results

|  | BA-shapes (Base) | Ba-shapes (motif) |
| ------------- | ------------- | ------------- |
| Accuracy (from Paper)  | 0.925 | 0.836 |
| Accuracy (from Replication)  | 0.971 | 0.911 |

<br/>

| | | | |
| ------------- | ------------- | ------------- | ------------- |
| Number of Nodes | 30 | 300 | 3000 |
| Accuracy | 0.965 | 0.971 | 0.99 |
<br/>

## Contributors

1. Aditya Chandrasekhar
2. Vincent Yong Wei Jie