

class Explainer():
     """Uses Shapley values to explain any machine learning model or python function.

    This is the primary explainer interface for the SHAP library. It takes any combination
    of a model and masker and returns a callable subclass object that implements
    the particular estimation algorithm that was chosen.
    """