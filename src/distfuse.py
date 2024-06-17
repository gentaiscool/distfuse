from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from typing import List, Any
import numpy as np

class DistFuse():
    """
        A DistFuse class to compute similarity scores from multiple models.
    
        e.g.,
            from distfuse import DistFuse

            model_checkpoints = ["sentence-transformers/LaBSE", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
            weights = [1, 1]
            dist_measure = "cosine"
            model = DistFuse(model_checkpoints, weights, dist_measure)
            
            scores = model(["I like apple", "I like cats"], ["I like orange", "I like dogs"])
            print(scores.shape)
    """
    def __init__(self, model_checkpoints:List[str], weights:List[float]=None, dist_measure:str="euclid"):
        """
            Args:
                model_checkpoints (List[str]): a list of model checkpoints
                weights (List[float]): a list of weights
                dist_measure (str): the distance measure
        """
        self.model_checkpoints = model_checkpoints
        self.models = []

        if dist_measure == "euclid":
            self.dist_measure = euclidean_distances
        elif dist_measure == "cosine":
            self.dist_measure = cosine_similarity
        elif dist_measure == "manhattan":
            self.dist_measure = manhattan_distances
        else:
            raise ValueError(f"dist_measure {dist_measure} is not found.")

        for i in range(len(self.model_checkpoints)):
            model_checkpoint = self.model_checkpoints[i]
            model = SentenceTransformer(model_checkpoint)
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
