"""Cython implementation of MALIS loss and graph operations.

This module provides efficient C++-backed implementations of:
- MALIS loss weight computation
- Connected components analysis
- Marker watershed segmentation
- Affinity graph operations
- Evaluation metrics (Rand index, V_rand)
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t
from scipy.special import comb
import scipy.sparse

cdef extern from "malis_cpp.h":
    void malis_loss_weights_cpp(const uint64_t nVert, const uint64_t* segTrue,
                   const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
                   const int pos,
                   uint64_t* nPairPerEdge);
    void connected_components_cpp(const uint64_t nVert,
                   const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const int* edgeWeight,
                   uint64_t* seg);
    void marker_watershed_cpp(const uint64_t nVert, const uint64_t* marker,
                   const uint64_t nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
                   uint64_t* seg);

def malis_loss_weights(
    np.ndarray[uint64_t, ndim=1] segTrue,
    np.ndarray[uint64_t, ndim=1] node1,
    np.ndarray[uint64_t, ndim=1] node2,
    np.ndarray[float, ndim=1] edgeWeight,
    int pos,
):
    """Compute MALIS loss weights for an edge list.

    Args:
        segTrue: Ground-truth segmentation labels for each vertex (uint64).
        node1: First node indices for each edge (uint64).
        node2: Second node indices for each edge (uint64).
        edgeWeight: Weight for each edge (float).
        pos: Pass type (1 for positive, 0 for negative).

    Returns:
        Number of pairs of nodes for each edge (uint64 array).
    """
    cdef uint64_t nVert = segTrue.shape[0]
    cdef uint64_t nEdge = node1.shape[0]
    segTrue = np.ascontiguousarray(segTrue)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t, ndim=1] nPairPerEdge = np.zeros(
        edgeWeight.shape[0], dtype=np.uint64
    )
    malis_loss_weights_cpp(
        nVert, &segTrue[0],
        nEdge, &node1[0], &node2[0], &edgeWeight[0],
        pos,
        &nPairPerEdge[0]
    )
    return nPairPerEdge


def connected_components(
    uint64_t nVert,
    np.ndarray[uint64_t, ndim=1] node1,
    np.ndarray[uint64_t, ndim=1] node2,
    np.ndarray[int, ndim=1] edgeWeight,
    int sizeThreshold=1,
):
    """Compute connected components from an edge list.

    Args:
        nVert: Number of vertices in the graph.
        node1: First node indices for each edge (uint64).
        node2: Second node indices for each edge (uint64).
        edgeWeight: Binary weight for each edge (int, 1=connected, 0=disconnected).
        sizeThreshold: Minimum component size to keep (default=1).

    Returns:
        Tuple of (segmentation, component_sizes) where segmentation assigns
        each vertex to a component ID, and component_sizes gives the size of
        each component.
    """
    cdef uint64_t nEdge = node1.shape[0]
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t, ndim=1] seg = np.zeros(nVert, dtype=np.uint64)
    connected_components_cpp(
        nVert,
        nEdge, &node1[0], &node2[0], &edgeWeight[0],
        &seg[0]
    )
    seg, segSizes = prune_and_renum(seg, sizeThreshold)
    return seg, segSizes


def marker_watershed(
    np.ndarray[uint64_t, ndim=1] marker,
    np.ndarray[uint64_t, ndim=1] node1,
    np.ndarray[uint64_t, ndim=1] node2,
    np.ndarray[float, ndim=1] edgeWeight,
    int sizeThreshold=1,
):
    """Perform marker-based watershed segmentation on a graph.

    Args:
        marker: Initial marker labels for each vertex (uint64).
        node1: First node indices for each edge (uint64).
        node2: Second node indices for each edge (uint64).
        edgeWeight: Weight for each edge (float).
        sizeThreshold: Minimum component size to keep (default=1).

    Returns:
        Tuple of (segmentation, component_sizes) where segmentation assigns
        each vertex to a watershed region, and component_sizes gives the size
        of each region.
    """
    cdef uint64_t nVert = marker.shape[0]
    cdef uint64_t nEdge = node1.shape[0]
    marker = np.ascontiguousarray(marker)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[uint64_t, ndim=1] seg = np.zeros(nVert, dtype=np.uint64)
    marker_watershed_cpp(
        nVert, &marker[0],
        nEdge, &node1[0], &node2[0], &edgeWeight[0],
        &seg[0]
    )
    seg, segSizes = prune_and_renum(seg, sizeThreshold)
    return seg, segSizes



def prune_and_renum(
    np.ndarray[uint64_t, ndim=1] seg,
    int sizeThreshold=1,
):
    """Renumber segmentation components and prune small ones.

    Renumbers components in descending order by size and optionally removes
    components below a size threshold.

    Args:
        seg: Segmentation array with component IDs (uint64).
        sizeThreshold: Minimum component size to keep (default=1).
            Components smaller than this are set to 0.

    Returns:
        Tuple of (renumbered_seg, component_sizes) where renumbered_seg has
        components numbered 1..N in descending size order, and component_sizes
        gives the size of each remaining component.
    """
    # Renumber the components in descending order by size
    segId, segSizes = np.unique(seg, return_counts=True)
    descOrder = np.argsort(segSizes)[::-1]
    renum = np.zeros(int(segId.max() + 1), dtype=np.uint64)
    segId = segId[descOrder]
    segSizes = segSizes[descOrder]
    renum[segId] = np.arange(1, len(segId) + 1)

    if sizeThreshold > 0:
        renum[segId[segSizes <= sizeThreshold]] = 0
        segSizes = segSizes[segSizes > sizeThreshold]

    seg = renum[seg]
    return seg, segSizes


def bmap_to_affgraph(bmap, nhood, return_min_idx=False):
    """Construct an affinity graph from a boundary map.

    Args:
        bmap: Boundary probability map with shape (z, y, x).
        nhood: Neighborhood offsets with shape (n_edges, 3).
        return_min_idx: Unused parameter for backward compatibility.

    Returns:
        Affinity graph with shape (n_edges, z, y, x), where each edge stores
        the minimum boundary probability along that edge.
    """
    # Assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = bmap.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)
    minidx = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e,
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = np.minimum(
                bmap[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                     max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])],
                bmap[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                     max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]
            )
        minidx[e,
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = (
                bmap[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                     max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] >
                bmap[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                     max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]
            )

    return aff

def seg_to_affgraph(seg, nhood):
    """Construct an affinity graph from a segmentation.

    Args:
        seg: Segmentation with shape (z, y, x).
        nhood: Neighborhood offsets with shape (n_edges, 3).

    Returns:
        Affinity graph with shape (n_edges, z, y, x), where each edge is 1
        if the two voxels belong to the same non-background segment, 0 otherwise.
    """
    # Assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e,
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = (
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                     max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] ==
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                     max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]) *
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                     max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] > 0) *
                (seg[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                     max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])] > 0)
            )

    return aff

def segmask_to_affmask(mask, nhood):
    """Construct an affinity mask from a binary segmentation mask.

    Args:
        mask: Binary mask with shape (z, y, x).
        nhood: Neighborhood offsets with shape (n_edges, 3).

    Returns:
        Affinity mask with shape (n_edges, z, y, x), where each edge is 1
        if both voxels are non-zero in the mask, 0 otherwise.
    """
    # Assume affinity mask is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = mask.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    for e in range(nEdge):
        aff[e,
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = (
                (mask[max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
                      max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
                      max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] > 0) *
                (mask[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                      max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                      max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])] > 0)
            )

    return aff

def nodelist_like(shape, nhood):
    """Construct node lists for edge list representation of an affinity graph.

    Args:
        shape: Shape of the volume (z, y, x).
        nhood: Neighborhood offsets with shape (n_edges, 3).

    Returns:
        Tuple of (node1, node2) where each is an array of shape
        (n_edges, z, y, x) containing the node indices for each edge.
    """
    # Assume node shape is represented as:
    # shape = (z, y, x)
    # nhood.shape = (edges, 3)
    nEdge = nhood.shape[0]
    nodes = np.arange(np.prod(shape), dtype=np.uint64).reshape(shape)
    node1 = np.tile(nodes, (nEdge, 1, 1, 1))
    node2 = np.full(node1.shape, -1, dtype=np.uint64)

    for e in range(nEdge):
        node2[e,
            max(0, -nhood[e, 0]):min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]):min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]):min(shape[2], shape[2] - nhood[e, 2])] = \
                nodes[max(0, nhood[e, 0]):min(shape[0], shape[0] + nhood[e, 0]),
                      max(0, nhood[e, 1]):min(shape[1], shape[1] + nhood[e, 1]),
                      max(0, nhood[e, 2]):min(shape[2], shape[2] + nhood[e, 2])]

    return node1, node2


def affgraph_to_edgelist(aff, nhood):
    """Convert affinity graph to edge list representation.

    Args:
        aff: Affinity graph with shape (n_edges, z, y, x).
        nhood: Neighborhood offsets with shape (n_edges, 3).

    Returns:
        Tuple of (node1, node2, edge_weights) as flattened 1D arrays.
    """
    node1, node2 = nodelist_like(aff.shape[1:], nhood)
    return node1.ravel(), node2.ravel(), aff.ravel()


def connected_components_affgraph(aff, nhood):
    """Compute connected components from an affinity graph.

    Args:
        aff: Affinity graph with shape (n_edges, z, y, x).
        nhood: Neighborhood offsets with shape (n_edges, 3).

    Returns:
        Tuple of (segmentation, component_sizes).
    """
    node1, node2, edge = affgraph_to_edgelist(aff, nhood)
    seg, segSizes = connected_components(int(np.prod(aff.shape[1:])), node1, node2, edge)
    seg = seg.reshape(aff.shape[1:])
    return seg, segSizes


def mk_cont_table(seg1, seg2):
    """Make a contingency table between two segmentations.

    Args:
        seg1: First segmentation (flattened).
        seg2: Second segmentation (flattened).

    Returns:
        Contingency table as a 2D array.
    """
    cont_table = scipy.sparse.coo_matrix((np.ones(seg1.shape), (seg1, seg2))).toarray()
    return cont_table


def compute_V_rand_N2(segTrue, segEst):
    """Compute variation of information based Rand scores.

    Computes V_rand, V_rand_split, and V_rand_merge metrics between
    ground-truth and estimated segmentations, ignoring background (label 0).

    Args:
        segTrue: Ground-truth segmentation.
        segEst: Estimated segmentation.

    Returns:
        Tuple of (V_rand, V_rand_split, V_rand_merge).
    """
    segTrue = segTrue.ravel()
    segEst = segEst.ravel()
    idx = segTrue != 0
    segTrue = segTrue[idx]
    segEst = segEst[idx]

    cont_table = scipy.sparse.coo_matrix((np.ones(segTrue.shape), (segTrue, segEst))).toarray()
    P = cont_table / cont_table.sum()
    t = P.sum(axis=0)
    s = P.sum(axis=1)

    V_rand_split = (P**2).sum() / (t**2).sum()
    V_rand_merge = (P**2).sum() / (s**2).sum()
    V_rand = 2 * (P**2).sum() / ((t**2).sum() + (s**2).sum())

    return V_rand, V_rand_split, V_rand_merge


def rand_index(segTrue, segEst):
    """Compute Rand index and related metrics.

    Computes Rand index, F-score, precision, and recall between ground-truth
    and estimated segmentations, ignoring background (label 0).

    Args:
        segTrue: Ground-truth segmentation.
        segEst: Estimated segmentation.

    Returns:
        Tuple of (rand_index, fscore, precision, recall).
    """
    segTrue = segTrue.ravel()
    segEst = segEst.ravel()
    idx = segTrue != 0
    segTrue = segTrue[idx]
    segEst = segEst[idx]

    tp_plus_fp = comb(np.bincount(segTrue), 2).sum()
    tp_plus_fn = comb(np.bincount(segEst), 2).sum()
    A = np.c_[(segTrue, segEst)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(segTrue))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    ri = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    fscore = 2 * prec * rec / (prec + rec)
    return ri, fscore, prec, rec


def mknhood2d(radius=1):
    """Create 2D neighborhood offsets for a circular neighborhood.

    Args:
        radius: Maximum distance from center (default=1).

    Returns:
        Array of shape (n_edges, 2) containing (y, x) offsets for edges
        within the specified radius.
    """
    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad + 1, 1)
    y = np.arange(-ceilrad, ceilrad + 1, 1)
    i, j = np.meshgrid(y, x)

    idxkeep = (i**2 + j**2) <= radius**2
    i = i[idxkeep].ravel()
    j = j[idxkeep].ravel()
    zeroIdx = np.ceil(len(i) / 2).astype(np.int32)

    nhood = np.vstack((i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))


def mknhood3d(radius=1):
    """Create 3D neighborhood offsets for a spherical neighborhood.

    The neighborhood reference for the dense graph representation we use:
    nhood[i,:] is a 3-vector describing the node that conn[:,:,:,i] connects to.
    So conn[z,y,x,i] is the edge between node [z,y,x] and [z,y,x]+nhood[i,:].
    In other words, nhood is just the offset vector that each edge corresponds to.

    Args:
        radius: Maximum distance from center (default=1).

    Returns:
        Array of shape (n_edges, 3) containing (z, y, x) offsets for edges
        within the specified radius.
    """
    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad + 1, 1)
    y = np.arange(-ceilrad, ceilrad + 1, 1)
    z = np.arange(-ceilrad, ceilrad + 1, 1)
    i, j, k = np.meshgrid(z, y, x)

    idxkeep = (i**2 + j**2 + k**2) <= radius**2
    i = i[idxkeep].ravel()
    j = j[idxkeep].ravel()
    k = k[idxkeep].ravel()
    zeroIdx = np.ceil(len(i) / 2).astype(np.int32)

    nhood = np.vstack((k[:zeroIdx], i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))


def mknhood3d_aniso(radiusxy=1, radiusxy_zminus1=1.8):
    """Create anisotropic 3D neighborhood for volumes with different z-resolution.

    Creates a neighborhood with:
    - Isotropic connectivity within the current z-plane (radius=radiusxy)
    - Extended connectivity to the previous z-plane (radius=radiusxy_zminus1)

    Args:
        radiusxy: Maximum distance in xy-plane within same z (default=1).
        radiusxy_zminus1: Maximum distance in xy-plane to z-1 plane (default=1.8).

    Returns:
        Array of shape (n_edges, 3) containing (z, y, x) offsets.
    """
    nhoodxyz = mknhood3d(radiusxy)
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)

    nhood = np.zeros((nhoodxyz.shape[0] + 2 * nhoodxy_zminus1.shape[0], 3), dtype=np.int32)
    nhood[:nhoodxyz.shape[0]] = nhoodxyz
    nhood[nhoodxyz.shape[0]:, 0] = -1
    nhood[nhoodxyz.shape[0]:, 1:] = np.vstack((nhoodxy_zminus1, -nhoodxy_zminus1))

    return np.ascontiguousarray(nhood)
