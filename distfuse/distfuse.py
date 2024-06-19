from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from typing import List, Any
import numpy as np
from openai import OpenAI
import cohere

class EmbeddingModel():
    """
        An embedding model class
    """
    def __init__(self, model_checkpoint:str, type:str="hf", openai_token:str="", cohere_token:str=""):
        self.model_checkpoint = model_checkpoint
        self.type = type

        if type == "openai":
            self.model = OpenAI(api_key=openai_token)
        elif type == "cohere":
            self.model = cohere.Client(cohere_token)
        elif type == "hf": # huggingface
            self.model = SentenceTransformer(model_checkpoint)
        else:
            raise ValueError(f"We only support openai, cohere, and hf as model_checkpoint type.")
    
    def get_openai_embedding(self, texts):
        data = self.model.embeddings.create(input = texts, model=self.model_checkpoint).data
        embeddings = []
        for obj in data:
            embeddings.append(obj.embedding)
        return embeddings

    def get_cohere_embedding(self, texts):
        response = self.model.embed(texts=texts, model=self.model_checkpoint, input_type="search_query")
        return response.embeddings

    def encode(self, texts):
        if self.type == "openai":
            embeddings = self.get_openai_embedding(texts)
        elif self.type == "cohere":
            embeddings = self.get_cohere_embedding(texts)
        else:
            embeddings = self.model.encode(texts)
        return embeddings


class DistFuse():
    """
        A DistFuse class to compute similarity scores from multiple models.
    
        e.g.,
            from distfuse import DistFuse

            model_checkpoints = [["sentence-transformers/LaBSE", "hf"], ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "hf"]]
            weights = [1, 1]
            dist_measure = "cosine"
            model = DistFuse(model_checkpoints, weights, dist_measure)
            
            scores = model.score_pairs(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
            print(scores)
    """
    def __init__(self, model_checkpoints:List[List[str]], weights:List[float]=None, dist_measure:str="euclid", openai_token=None, cohere_token=None):
        """
            Args:
                model_checkpoints (List[str]): a list of model checkpoints and types
                weights (List[float]): a list of weights
                dist_measure (str): the distance measure (only accept euclidean, cosine, manhattan, by default: euclidean)
                openai_token (str): openai token
                cohere_token (str): cohere token
        """
        self.model_checkpoints = model_checkpoints
        self.models = []

        if dist_measure == "euclidean":
            self.dist_measure = euclidean_distances
        elif dist_measure == "cosine":
            self.dist_measure = cosine_similarity
        elif dist_measure == "manhattan":
            self.dist_measure = manhattan_distances
        else:
            raise ValueError(f"dist_measure {dist_measure} is not found.")

        for i in range(len(self.model_checkpoints)):
            model_checkpoint = self.model_checkpoints[i]
            model = EmbeddingModel(model_checkpoint[0], type=model_checkpoint[1], openai_token=openai_token, cohere_token=cohere_token)
            self.models.append(model)

        if weights is not None:
            self.weights = [1] * len(self.models)
        else:
            self.weights = weights

        assert len(self.models) == len(self.weights)

    def score_pairs(self, text_list1:List[str], text_list2:List[str]) -> List[float]:
        """
            Compute the scores of two text sequence lists
            Args:
                text_list1 (List[str]): a list of text sequences (m samples)
                text_list2 (List[str]): a list of text sequences (n samples)
            Returns:
                List[float]: a list of scores (m x n dimensions)
        """
        
        assert len(text_list1) > 0 and len(text_list2) > 0

        scores = []
        for model in self.models:
            embs1 = model.encode(text_list1)
            embs2 = model.encode(text_list2)
            scores.append(self.dist_measure(embs1, embs2))

        final_scores = scores[0]
        for i in range(1, len(scores)):
            final_scores = final_scores + scores[i]
        return final_scores
