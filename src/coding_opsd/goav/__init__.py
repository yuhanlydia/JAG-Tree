"""Gradient-optimal active verification reference operators."""

from .estimator import (
    aipw_gradient,
    aipw_pseudolabel,
    design_risk,
    direct_loo_gradient,
    exact_realized_design_mse,
    loo_influence,
)
from .noise import oracle_joint_posterior, simulate_synthetic_oracle
from .solver import bernoulli_design, full_audit_design, poisson_neyman_design, score_design, solve_goav
from .subsets import SubsetDesign, enumerate_subsets, inclusion_probabilities, sample_subset

__all__ = [
    "SubsetDesign",
    "aipw_gradient",
    "aipw_pseudolabel",
    "bernoulli_design",
    "design_risk",
    "direct_loo_gradient",
    "enumerate_subsets",
    "exact_realized_design_mse",
    "full_audit_design",
    "inclusion_probabilities",
    "loo_influence",
    "oracle_joint_posterior",
    "poisson_neyman_design",
    "sample_subset",
    "score_design",
    "simulate_synthetic_oracle",
    "solve_goav",
]
