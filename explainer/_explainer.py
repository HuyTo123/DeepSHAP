import copy
import time

import numpy as np
import pandas as pd
import scipy.sparse

from .. import explainers, links, maskers, models
from .._explanation import Explanation
from .._serializable import Deserializer, Serializable, Serializer
from ..maskers import Masker
from ..models import Model



class Explainer(Serializable):
    """Uses Shapley values to explain any machine learning model or python function.

    This is the primary explainer interface for the SHAP library. It takes any combination
    of a model and masker and returns a callable subclass object that implements
    the particular estimation algorithm that was chosen.
    """

    def __init__(
        self,
        model,
        masker=None,
        link=links.identity,
        algorithm="auto",
        output_names=None,
        feature_names=None,
        linearize_link=True,
        seed=None,
        **kwargs,
    ):
        """Build a new explainer for the passed model.

        Parameters
        ----------
        model : object or function
            User supplied function or model object that takes a dataset of samples and
            computes the output of the model for those samples.

        masker : function, numpy.array, pandas.DataFrame, tokenizer, None, or a list of these for each model input
            The function used to "mask" out hidden features of the form `masked_args = masker(*model_args, mask=mask)`.
            It takes input in the same form as the model, but for just a single sample with a binary
            mask, then returns an iterable of masked samples. These
            masked samples will then be evaluated using the model function and the outputs averaged.
            As a shortcut for the standard masking using by SHAP you can pass a background data matrix
            instead of a function and that matrix will be used for masking. Domain specific masking
            functions are available in shap such as shap.ImageMasker for images and shap.TokenMasker
            for text. In addition to determining how to replace hidden features, the masker can also
            constrain the rules of the cooperative game used to explain the model. For example
            shap.TabularMasker(data, hclustering="correlation") will enforce a hierarchical clustering
            of coalitions for the game (in this special case the attributions are known as the Owen values).

        link : function
            The link function used to map between the output units of the model and the SHAP value units. By
            default it is shap.links.identity, but shap.links.logit can be useful so that expectations are
            computed in probability units while explanations remain in the (more naturally additive) log-odds
            units. For more details on how link functions work see any overview of link functions for generalized
            linear models.

        algorithm : "auto", "permutation", "partition", "tree", or "linear"
            The algorithm used to estimate the Shapley values. There are many different algorithms that
            can be used to estimate the Shapley values (and the related value for constrained games), each
            of these algorithms have various tradeoffs and are preferable in different situations. By
            default the "auto" options attempts to make the best choice given the passed model and masker,
            but this choice can always be overridden by passing the name of a specific algorithm. The type of
            algorithm used will determine what type of subclass object is returned by this constructor, and
            you can also build those subclasses directly if you prefer or need more fine grained control over
            their options.

        output_names : None or list of strings
            The names of the model outputs. For example if the model is an image classifier, then output_names would
            be the names of all the output classes. This parameter is optional. When output_names is None then
            the Explanation objects produced by this explainer will not have any output_names, which could effect
            downstream plots.

        seed: None or int
            seed for reproducibility

        """
        self.model = model
        self.output_names = output_names
        self.feature_names = feature_names

        # wrap the incoming masker object as a shap.Masker object
        if isinstance(masker, pd.DataFrame) or (
            (isinstance(masker, np.ndarray) or scipy.sparse.issparse(masker)) and len(masker.shape) == 2
        ):
            if algorithm == "partition":
                self.masker = maskers.Partition(masker)
            else:
                self.masker = maskers.Independent(masker)
        elif (masker is list or masker is tuple) and masker[0] is not str:
            self.masker = maskers.Composite(*masker)
        elif (masker is dict) and ("mean" in masker):
            self.masker = maskers.Independent(masker)
        else:
            self.masker = masker

        # Check for transformer pipeline objects and wrap them
        

        # self._brute_force_fallback = explainers.BruteForce(self.model, self.masker)

        # validate and save the link function
        if callable(link):
            self.link = link
        else:
            raise TypeError("The passed link function needs to be callable!")
        self.linearize_link = linearize_link

        # if we are called directly (as opposed to through super()) then we convert ourselves to the subclass
        # that implements the specific algorithm that was chosen
        # if we call from DeepExplainer we don't need this below code 
        if self.__class__ is Explainer:         
            # build the right subclass
            if algorithm == "deep":
                self.__class__ = explainers.DeepExplainer
                explainers.DeepExplainer.__init__(
                    self,
                    self.model,
                    self.masker,
                    link=self.link,
                    feature_names=self.feature_names,
                    linearize_link=linearize_link,
                    **kwargs,
                )
            else:
                raise Exception(f"Unknown algorithm type passed: {algorithm}!")

  
    def save(self, out_file, model_saver=".save", masker_saver=".save"):
        """Write the explainer to the given file stream."""
        super().save(out_file)
        with Serializer(out_file, "shap.Explainer", version=0) as s:
            s.save("model", self.model, model_saver)
            s.save("masker", self.masker, masker_saver)
            s.save("link", self.link)

    @classmethod
    def load(cls, in_file, model_loader=Model.load, masker_loader=Masker.load, instantiate=True):
        """Load an Explainer from the given file stream.

        Parameters
        ----------
        in_file : The file stream to load objects from.

        """
        if instantiate:
            return cls._instantiated_load(in_file, model_loader=model_loader, masker_loader=masker_loader)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.Explainer", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model", model_loader)
            kwargs["masker"] = s.load("masker", masker_loader)
            kwargs["link"] = s.load("link")
        return kwargs


def pack_values(values):
    """Used the clean up arrays before putting them into an Explanation object."""
    if not hasattr(values, "__len__"):
        return values

    # collapse the values if we didn't compute them
    if values is None or values[0] is None:
        return None

    # convert to a single numpy matrix when the array is not ragged
    elif np.issubdtype(type(values[0]), np.number) or len(np.unique([len(v) for v in values])) == 1:
        return np.array(values)
    else:
        return np.array(values, dtype=object)