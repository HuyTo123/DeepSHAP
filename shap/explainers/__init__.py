from ._deep import DeepExplainer
from ._gradient import GradientExplainer


# Alternative legacy "short-form" aliases, which are kept here for backwards-compatibility

Deep = DeepExplainer
Gradient = GradientExplainer


__all__ = [
    "DeepExplainer",
    "GradientExplainer",
]
