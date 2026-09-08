"""Controlled allocation-arm scoring equations."""

from __future__ import annotations

import numpy as np

from .allocator import AllocationNode, JointMoments, JointTraceMoments, joint_risk, joint_risk_no_cross
from .registry import ALLOCATOR_ARMS


def _trace(moments: JointMoments | JointTraceMoments) -> float:
    if isinstance(moments, JointTraceMoments):
        return float(moments.gradient_variance_trace)
    return float(np.trace(moments.gradient_covariance))


def arm_risk(node: AllocationNode, moments: JointMoments | JointTraceMoments, arm: str) -> float:
    if arm not in ALLOCATOR_ARMS:
        raise ValueError(f"unknown allocator arm: {arm!r}")
    if arm == "uniform":
        return 1.0
    if arm == "entropy":
        return float(node.entropy)
    if arm == "value_variance":
        return float(moments.value_variance)
    if arm == "gradient_only":
        return _trace(moments)
    if arm == "trace_score":
        return float(node.trace_score)
    if arm == "jag_no_cross":
        return joint_risk_no_cross(moments, node.ancestor_score)
    return joint_risk(moments, node.ancestor_score)
