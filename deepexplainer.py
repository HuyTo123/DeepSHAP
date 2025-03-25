from .pytorchdeep import PyTorchDeep
from .explanation import Explanation
from ._explainer import Explainer
import torch as torch
class DeepExplainer(Explainer):
    """"deepLIFT explainer
        Model is the model will use to explain
        Data is the data we want to explain, you should only use something like 100 or 1000 random 
        background samples to explain the model.
        Session is the session we want to run the model in
        Learning_phase_flags is the flags we want to set for the model
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        # First, we need to find the framework
        if type(model) is tuple:
            a, b = model
            try:
                a.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'
        else:
            try:
                model.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'
        #create self.explainer
        if framework == 'pytorch':
            self.explainer = PyTorchDeep(model, masker)
        else:
            raise Exception("Invalid framework")
        # masker is the pre-processing function for the input data
        masker = data
        super()
        self.expected_value = self.explainer.expected_value
        self.explainer.framework = framework
    """ Return value -> Explanation, use call for covinience 
        obj = className()
        obj(parameters) -> obj.__call__(parameters)
    """
    def __call__(self, X: torch.tensor) -> Explanation:  # type: ignore  # noqa: F821
        """Return an explanation object for the model applied to X.

        Parameters
        ----------
        X : if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        Returns
        -------
        shap.Explanation:
        """
        shap_values = self.shap_values(X)
        return Explanation(values=shap_values, data=X)

    def shap_values(self, X: torch.tensor, ranked_outputs=None, output_rank_order="max", check_additivity=True): 
        """Return approximate SHAP values for the model applied to the data given by X.

        Parameters
        ----------
        X : if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where shap_values is a list of numpy
            arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
            which output indexes were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        Returns
        -------
        np.array or list
            Estimated SHAP values, usually of shape ``(# samples x # features)``.

            The shape of the returned array depends on the number of model outputs:

            * one input, one output: matrix of shape ``(#num_samples, *X.shape[1:])``.
            * one input, multiple outputs: matrix of shape ``(#num_samples, *X.shape[1:], #num_outputs)``
            * multiple inputs, one or more outputs: list of matrices, with shapes of one of the above.

            If ranked_outputs is ``None`` then this list of tensors matches
            the number of model outputs. If ranked_outputs is a positive integer a pair is returned
            (shap_values, indexes), where shap_values is a list of tensors with a length of
            ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
            were chosen as "top".

            .. versionchanged:: 0.45.0
                Return type for models with multiple outputs and one input changed from list to np.ndarray.

        """
        return self.explainer.shap_values(X, ranked_outputs, output_rank_order, check_additivity=check_additivity)