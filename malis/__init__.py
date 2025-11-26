"""MALIS (MAximum LIkelihood Segmentation) loss function package.

This package provides functions for computing the MALIS loss for supervised
learning of image segmentation and clustering.
"""

from .malis import (
    bmap_to_affgraph,
    compute_V_rand_N2,
    connected_components,
    connected_components_affgraph,
    malis_loss_weights,
    marker_watershed,
    mk_cont_table,
    mknhood2d,
    mknhood3d,
    mknhood3d_aniso,
    nodelist_like,
    prune_and_renum,
    rand_index,
    seg_to_affgraph,
    segmask_to_affmask,
)
from .malis_loss import malis_loss_op, malis_weights_op, MalisWeights

__all__ = [
    # Core MALIS functions
    "malis_loss_weights",
    "malis_weights_op",
    "malis_loss_op",
    "MalisWeights",
    # Affinity graph functions
    "seg_to_affgraph",
    "segmask_to_affmask",
    "bmap_to_affgraph",
    "nodelist_like",
    # Connected components functions
    "connected_components",
    "connected_components_affgraph",
    "marker_watershed",
    "prune_and_renum",
    # Neighborhood functions
    "mknhood2d",
    "mknhood3d",
    "mknhood3d_aniso",
    # Evaluation metrics
    "rand_index",
    "compute_V_rand_N2",
    "mk_cont_table",
]
