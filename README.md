# DistFuse

DistFuse is a library to calculate similarity scores between two collections of text sequences encoded using transformer models. This library allows combining more than one models, including APIs,  

## Table of Contents

- [Install](#install)
- [Reference](#reference)
- [Usage](#usage)

## Install
```
pip install distfuse
```

## Reference
If you use any source codes included in this toolkit in your work, please cite the following papers [1](https://arxiv.org/pdf/2406.07424) [2](https://aclanthology.org/2023.ijcnlp-short.11.pdf).
```
@article{winata2024miners,
  title={MINERS: Multilingual Language Models as Semantic Retrievers},
  author={Winata, Genta Indra and Zhang, Ruochen and Adelani, David Ifeoluwa},
  journal={arXiv preprint arXiv:2406.07424},
  year={2024}
}
@inproceedings{winata2023efficient,
  title={Efficient Zero-Shot Cross-lingual Inference via Retrieval},
  author={Winata, Genta and Xie, Lingjue and Radhakrishnan, Karthik and Gao, Yifan and Preo{\c{t}}iuc-Pietro, Daniel},
  booktitle={Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={93--104},
  year={2023}
}
```

## Usage
We support `hf` (Hugging Face models), and APIs, such as `cohere`, and `openai`. 

DistFuse with 2 models.
```python
from distfuse import DistFuse

model_checkpoints = [["sentence-transformers/LaBSE", "hf"], ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "hf"]]
weights = [1, 1]
dist_measure = "cosine" # cosine, euclidean, manhattan
model = DistFuse(model_checkpoints, weights, dist_measure)

scores = model.score_pairs(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
print(scores)
```

DistFuse with 3 models.
```python
from distfuse import DistFuse

model_checkpoints = [["sentence-transformers/LaBSE", "hf"], ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "hf"], ["text-embedding-3-large", "openai"]]
weights = [1, 1, 1]
dist_measure = "cosine"
model = DistFuse(model_checkpoints, weights, dist_measure, openai_token="", cohere_token="")

scores = model.score_pairs(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
print(scores)
```
