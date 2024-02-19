# Learning Relation Prototype from Unlabeled Texts for Long-tail Relation Extraction

![Overview of Our proposed model](./figure/framework.png)





This repository provides the code for our TKDE paper: *Learning Relation Prototype from Unlabeled Texts for Long-tail Relation Extraction*.

## Dependencies

```
tqdm==4.46.0
numpy==1.18.5
tensorflow_gpu==1.15.0
matplotlib==3.3.3
scikit_learn==0.24.2
```



## Quick Start

You can run the experiments in just two steps:

### Download dataset

We use  [Riedel NYT](http://iesl.cs.umass.edu/riedel/ecml/) and [Google Distant Supervision (GDS)](https://arxiv.org/pdf/1804.06987.pdf) dataset for evaluation. We have uploaded the preprocessed dataset and the pretrained files on [Google Driver](https://drive.google.com/file/d/1b6a7Rzf0GGfvwyhlv-OTyd4lqfKkdvAD/view?usp=sharing). Download the [data.zip](https://drive.google.com/file/d/1b6a7Rzf0GGfvwyhlv-OTyd4lqfKkdvAD/view?usp=sharing) and uncompress it to `algorithm/`

### Run experiments

For training the relation extraction model, run the following command:

```shell
python3 train.py {DATASET_NAME} {ENCODER} {SELECTOR} {EXP}
```

Where the `DATASET_NAME` can be `nyt` or `gids` , the  `ENCODER` can be `pcnn` ,`cnn`  or `rnn`, the  `SELECTOR` can be `ave(means average pooling)` or `att(means selective attention)`, and the `EXP` can be `trp/tmr/none`, which means using relation prototype,  implicit mutual relation or nothing, more details please refer to out paper. As illustrated in our paper, the model achived best performance is PA-TRP, which can be trained by the following command:

```shell
python3 train.py nyt pcnn att trp
```

For testing, run:

```
python3 test.py {DATASET_NAME} {MODEL_NAME}
```



## Result

![](figure/result.jpeg)

## Citation
Please cite the following papers if you use this code in your work.
```
@ARTICLE{9483677,
  author={Cao, Yixin and Kuang, Jun and Gao, Ming and Zhou, Aoying and Wen, Yonggang and Chua, Tat-Seng},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Learning Relation Prototype from Unlabeled Texts for Long-tail Relation Extraction}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TKDE.2021.3096200}}
```
```
@INPROCEEDINGS {9101658,
author = {J. Kuang and Y. Cao and J. Zheng and X. He and M. Gao and A. Zhou},
booktitle = {2020 IEEE 36th International Conference on Data Engineering (ICDE)},
title = {Improving Neural Relation Extraction with Implicit Mutual Relations},
year = {2020},
volume = {},
issn = {},
pages = {1021-1032},
abstract = {Relation extraction (RE) aims at extracting the relation between two entities from the text corpora. It is a crucial task for Knowledge Graph (KG) construction. Most existing methods predict the relation between an entity pair by learning the relation from the training sentences, which contain the targeted entity pair. In contrast to existing distant supervision approaches that suffer from insufficient training corpora to extract relations, our proposal of mining implicit mutual relation from the massive unlabeled corpora transfers the semantic information of entity pairs into the RE model, which is more expressive and semantically plausible. After constructing an entity proximity graph based on the implicit mutual relations, we preserve the semantic relations of entity pairs via embedding each vertex of the graph into a low-dimensional space. As a result, we can easily and flexibly integrate the implicit mutual relations and other entity information, such as entity types, into the existing RE methods.Our experimental results on a New York Times and another Google Distant Supervision datasets suggest that our proposed neural RE framework provides a promising improvement for the RE task, and significantly outperforms the state-of-the-art methods. Moreover, the component for mining implicit mutual relations is so flexible that can help to improve the performance of both CNN-based and RNN-based RE models significant.},
keywords = {data mining;training;neural networks;noise measurement;task analysis;training data;semantics},
doi = {10.1109/ICDE48307.2020.00093},
url = {https://doi.ieeecomputersociety.org/10.1109/ICDE48307.2020.00093},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {apr}
}

```


## Reference

* [OpenNRE](https://github.com/thunlp/OpenNRE/tree/tensorflow)
