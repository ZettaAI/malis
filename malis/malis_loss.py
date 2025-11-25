"""TensorFlow operations for computing MALIS loss weights."""

try:
    import tensorflow as tf
except ImportError:
    tf = None

import numpy as np

from .malis import malis_loss_weights, nodelist_like


class MalisWeights:
    """Compute MALIS loss weights for affinity graphs.

    This class computes edge weights for the MALIS loss function based on
    predicted and ground-truth affinities and segmentation.

    Attributes:
        output_shape: Shape of the output volume as numpy array.
        neighborhood: Array of spatial offsets defining edge connectivity.
        edge_list: Precomputed node list for the affinity graph.
    """

    def __init__(self, output_shape, neighborhood):
        """Initialize MalisWeights.

        Args:
            output_shape: Shape of the output volume (z, y, x).
            neighborhood: Array of spatial offsets (n_edges, 3).
        """
        self.output_shape = np.asarray(output_shape)
        self.neighborhood = np.asarray(neighborhood)
        self.edge_list = nodelist_like(self.output_shape, self.neighborhood)

    def get_edge_weights(self, affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled):
        """Compute MALIS edge weights by combining positive and negative passes.

        Args:
            affs: Predicted affinities (n_edges, z, y, x).
            gt_affs: Ground-truth affinities (n_edges, z, y, x).
            gt_seg: Ground-truth segmentation (z, y, x).
            gt_aff_mask: Binary mask for known affinities (n_edges, z, y, x).
            gt_seg_unlabelled: Binary mask for unlabelled regions (z, y, x).

        Returns:
            Combined edge weights from positive and negative passes.
        """
        # Replace the unlabelled-object area with a new unique ID
        if gt_seg_unlabelled.size > 0:
            gt_seg[gt_seg_unlabelled == 0] = gt_seg.max() + 1

        assert affs.shape[0] == len(self.neighborhood)

        weights_neg = self.malis_pass(affs, gt_affs, gt_seg, gt_aff_mask, pos=0)
        weights_pos = self.malis_pass(affs, gt_affs, gt_seg, gt_aff_mask, pos=1)

        return weights_neg + weights_pos

    def malis_pass(self, affs, gt_affs, gt_seg, gt_aff_mask, pos):
        """Perform a single MALIS pass (positive or negative).

        Args:
            affs: Predicted affinities (n_edges, z, y, x).
            gt_affs: Ground-truth affinities (n_edges, z, y, x).
            gt_seg: Ground-truth segmentation (z, y, x).
            gt_aff_mask: Binary mask for known affinities (n_edges, z, y, x).
            pos: Pass type (1 for positive, 0 for negative).

        Returns:
            Normalized edge weights for this pass.
        """
        # Create a copy of the affinities and change them, such that in the
        #   positive pass (pos == 1): affs[gt_affs == 0] = 0
        #   negative pass (pos == 0): affs[gt_affs == 1] = 1
        pass_affs = np.copy(affs)

        if gt_aff_mask.size == 0:
            constraint_edges = gt_affs == (1 - pos)
        else:
            constraint_edges = np.logical_and(
                gt_affs == (1 - pos),
                gt_aff_mask == 1
            )
        pass_affs[constraint_edges] = (1 - pos)

        weights = malis_loss_weights(
            gt_seg.astype(np.uint64).flatten(),
            self.edge_list[0].flatten(),
            self.edge_list[1].flatten(),
            pass_affs.astype(np.float32).flatten(),
            pos
        )

        weights = weights.reshape((-1,) + tuple(self.output_shape))
        assert weights.shape[0] == len(self.neighborhood)

        # '1-pos' samples don't contribute in the 'pos' pass
        weights[gt_affs == (1 - pos)] = 0

        # Masked-out samples don't contribute
        if gt_aff_mask.size > 0:
            weights[gt_aff_mask == 0] = 0

        # Normalize
        weights = weights.astype(np.float32)
        num_pairs = np.sum(weights)
        if num_pairs > 0:
            weights = weights / num_pairs

        return weights

def malis_weights_op(
    affs,
    gt_affs,
    gt_seg,
    neighborhood,
    gt_aff_mask=None,
    gt_seg_unlabelled=None,
    name=None,
):
    """Return a TensorFlow op to compute the MALIS loss weights.

    This is to be multiplied with an edge-wise base loss and summed up to create
    the final loss. For the Euclidean loss, use ``malis_loss_op``.

    Args:
        affs: The predicted affinities (Tensor).
        gt_affs: The ground-truth affinities (Tensor).
        gt_seg: The corresponding segmentation to the ground-truth affinities
            (Tensor). Label 0 denotes background.
        neighborhood: A list of spatial offsets, defining the neighborhood for
            each voxel (Tensor).
        gt_aff_mask: A binary mask indicating where ground-truth affinities are
            known (known = 1, unknown = 0). This is to be used for sparsely
            labelled ground-truth. Edges with unknown affinities will not be
            constrained in the two MALIS passes, and will not contribute to the
            loss. Optional (Tensor).
        gt_seg_unlabelled: A binary mask indicating where the ground-truth
            contains unlabelled objects (labelled = 1, unlabelled = 0). This is
            to be used for ground-truth where only some objects have been
            labelled. Note that this mask is a complement to ``gt_aff_mask``:
            It is assumed that no objects cross from labelled to unlabelled,
            i.e., the boundary is a real object boundary. Ground-truth
            affinities within the unlabelled areas should be masked out in
            ``gt_aff_mask``. Ground-truth affinities between labelled and
            unlabelled areas should be zero in ``gt_affs``. Optional (Tensor).
        name: A name to use for the operators created. Optional.

    Returns:
        A tensor with the shape of ``affs``, with MALIS weights stored for each
        edge.
    """

    if gt_aff_mask is None:
        gt_aff_mask = tf.zeros((0,))
    if gt_seg_unlabelled is None:
        gt_seg_unlabelled = tf.zeros((0,))

    output_shape = gt_seg.get_shape().as_list()

    malis_weights = MalisWeights(output_shape, neighborhood)

    def malis_functor(affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled):
        return malis_weights.get_edge_weights(
            affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled
        )

    # Use tf.py_function for TensorFlow 2.x compatibility (replaces deprecated tf.py_func)
    try:
        # TensorFlow 2.x
        weights = tf.py_function(
            malis_functor,
            [affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled],
            tf.float32,
            name=name,
        )
        weights.set_shape(affs.shape)
    except AttributeError:
        # TensorFlow 1.x fallback
        weights = tf.py_func(
            malis_functor,
            [affs, gt_affs, gt_seg, gt_aff_mask, gt_seg_unlabelled],
            [tf.float32],
            name=name,
        )
        weights = weights[0]

    return weights

def malis_loss_op(
    affs,
    gt_affs,
    gt_seg,
    neighborhood,
    gt_aff_mask=None,
    gt_seg_unlabelled=None,
    name=None,
):
    """Return a TensorFlow op to compute the constrained MALIS loss.

    Uses the squared distance to the target values for each edge as base loss.

    In the simplest case, you need to provide predicted affinities (``affs``),
    ground-truth affinities (``gt_affs``), a ground-truth segmentation
    (``gt_seg``), and the neighborhood that corresponds to the affinities.

    This loss also supports masks indicating unknown ground-truth. We
    distinguish two types of unknowns:

        1. Out of ground-truth. This is the case at the boundary of your
           labelled area. It is unknown whether objects continue or stop at the
           transition of the labelled area. This mask is given on edges as
           argument ``gt_aff_mask``.

        2. Unlabelled objects. It is known that there exists a boundary between
           the labelled area and unlabelled objects. Within the unlabelled
           objects area, it is unknown where boundaries are. This mask is also
           given on edges as argument ``gt_aff_mask``, and with an additional
           argument ``gt_seg_unlabelled`` to indicate where unlabelled objects
           are in the ground-truth segmentation.

    Both types of unknowns require masking edges to exclude them from the loss:
    For "out of ground-truth", these are all edges that have at least one node
    inside the "out of ground-truth" area. For "unlabelled objects", these are
    all edges that have both nodes inside the "unlabelled objects" area.

    Args:
        affs: The predicted affinities (Tensor).
        gt_affs: The ground-truth affinities (Tensor).
        gt_seg: The corresponding segmentation to the ground-truth affinities
            (Tensor). Label 0 denotes background.
        neighborhood: A list of spatial offsets, defining the neighborhood for
            each voxel (Tensor).
        gt_aff_mask: A binary mask indicating where ground-truth affinities are
            known (known = 1, unknown = 0). This is to be used for sparsely
            labelled ground-truth and at the borders of labelled areas. Edges
            with unknown affinities will not be constrained in the two MALIS
            passes, and will not contribute to the loss. Optional (Tensor).
        gt_seg_unlabelled: A binary mask indicating where the ground-truth
            contains unlabelled objects (labelled = 1, unlabelled = 0). This is
            to be used for ground-truth where only some objects have been
            labelled. Note that this mask is a complement to ``gt_aff_mask``:
            It is assumed that no objects cross from labelled to unlabelled,
            i.e., the boundary is a real object boundary. Ground-truth
            affinities within the unlabelled areas should be masked out in
            ``gt_aff_mask``. Ground-truth affinities between labelled and
            unlabelled areas should be zero in ``gt_affs``. Optional (Tensor).
        name: A name to use for the operators created. Optional.

    Returns:
        A tensor with one element, the MALIS loss.
    """

    weights = malis_weights_op(
        affs,
        gt_affs,
        gt_seg,
        neighborhood,
        gt_aff_mask,
        gt_seg_unlabelled,
        name,
    )
    edge_loss = tf.square(gt_affs - affs)

    return tf.reduce_sum(tf.multiply(weights, edge_loss))
