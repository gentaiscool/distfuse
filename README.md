# DistFuse

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
```python
from distfuse import DistFuse

model_checkpoints = ["sentence-transformers/LaBSE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
weights = [1, 1]
dist_measure = "cosine"
model = DistFuse(model_checkpoints, weights, dist_measure)

scores = model(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
print(scores.shape)
```
