# DistFuse

DistFuse is a library to calculate similarity scores between two collections of text sequences encoded using transformer models. This library allows combining more than one models, including APIs from Cohere and OpenAI.

## Table of Contents

- [Install](#install)
- [Reference](#reference)
- [Usage](#usage)
- [How to Contribute?](#-how-to-contribute)

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
We support `hf` (Hugging Face models), and APIs, such as `cohere`, and `openai`. For `dist_measure`, we support `cosine`, `euclidean`, and `manhattan`. If you are planning to use API models, please pass the appropriate token to `openai_token` or `cohere_token`. To use more than one model, add the model information to `model_checkpoints` and the weight to `weights`. There is no limit to the number of models you can use.

e.g., DistFuse with 2 models.
```python
from distfuse import DistFuse

model_checkpoints = [["sentence-transformers/LaBSE", "hf"], ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "hf"]]
weights = [1, 1]
dist_measure = "cosine" # cosine, euclidean, manhattan
model = DistFuse(model_checkpoints, weights, dist_measure, openai_token="", cohere_token="")

scores = model.score_pairs(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
print(scores)
```

e.g., DistFuse with 3 models. 
```python
from distfuse import DistFuse

model_checkpoints = [["sentence-transformers/LaBSE", "hf"], ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "hf"], ["text-embedding-3-large", "openai"]]
weights = [1, 1, 1]
dist_measure = "cosine"
model = DistFuse(model_checkpoints, weights, dist_measure, openai_token="", cohere_token="")

scores = model.score_pairs(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
print(scores)
```

## 🚀 How to Contribute?
Feel free to create [an issue](https://github.com/gentaiscool/distfuse/issues/) if you have any questions. And, create [a PR](https://github.com/gentaiscool/distfuse/pulls) for fixing bugs or adding improvements. 

If you are interested to create an extension of this work, feel free to reach out to [us](mailto:gentaindrawinata@gmail.com)!

Support our open source effort ⭐
