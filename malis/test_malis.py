"""Test suite for MALIS loss function.

This module contains tests for the MALIS segmentation loss function,
including neighborhood generation, connected components, and evaluation metrics.
"""

import datetime
import os

import h5py
import numpy as np

import malis as m

np.set_printoptions(precision=4)


def test_neighborhood_isotropic_3d():
    """Test creation of isotropic 3D neighborhood for 6-connected graph."""
    print("Can we make the `nhood' for an isotropic 3d dataset")
    print("corresponding to a 6-connected neighborhood?")
    nhood = m.mknhood3d(1)
    print(nhood)
    return nhood


def test_neighborhood_anisotropic_3d():
    """Test creation of anisotropic 3D neighborhood."""
    print("Can we make the `nhood' for an anisotropic 3d dataset")
    print("corresponding to a 4-connected neighborhood in-plane")
    print("and 26-connected neighborhood in the previous z-plane?")
    nhood = m.mknhood3d_aniso(1, 1.8)
    print(nhood)
    return nhood


def test_malis_loss_weights_basic():
    """Test basic MALIS loss weights computation on synthetic data."""
    seg_true = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5], dtype=np.uint64)
    node1 = np.arange(seg_true.shape[0] - 1, dtype=np.uint64)
    node2 = np.arange(1, seg_true.shape[0], dtype=np.uint64)
    n_vert = seg_true.shape[0]
    edge_weight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 2.5], dtype=np.float32)
    edge_weight = edge_weight / edge_weight.max()

    print(seg_true)
    print(edge_weight)

    n_pair_pos = m.malis_loss_weights(seg_true, node1, node2, edge_weight, 1)
    n_pair_neg = m.malis_loss_weights(seg_true, node1, node2, edge_weight, 0)
    print(np.vstack((n_pair_pos, n_pair_neg)))

    return n_pair_pos, n_pair_neg


def test_connected_components_basic():
    """Test connected components computation on synthetic data."""
    seg_true = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5], dtype=np.uint64)
    node1 = np.arange(seg_true.shape[0] - 1, dtype=np.uint64)
    node2 = np.arange(1, seg_true.shape[0], dtype=np.uint64)
    n_vert = seg_true.shape[0]
    edge_weight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 2.5], dtype=np.float32)
    edge_weight = edge_weight / edge_weight.max()

    idx_keep = (edge_weight > 0).astype(np.int32)
    cc = m.connected_components(n_vert, node1, node2, idx_keep)
    print(cc)
    return cc


def test_full_volume_processing(datadir=None):
    """Test MALIS processing on a full 3D volume.

    Args:
        datadir: Path to directory containing test data. If None, uses default path.
    """
    if datadir is None:
        datadir = '/groups/turaga/turagalab/greentea/project_data/dataset_06/fibsem_medulla_7col/tstvol-520-1-h5/'

    hdf5_gt_file = os.path.join(datadir, 'groundtruth_seg.h5')

    if not os.path.exists(hdf5_gt_file):
        print(f"Test data not found at {hdf5_gt_file}, skipping full volume test")
        return None, None, None, None

    now = datetime.datetime.now()
    print(f"[{now}] Reading test volume from {datadir}")

    with h5py.File(hdf5_gt_file, 'r') as h5seg:
        seg = np.asarray(h5seg['main']).astype(np.int32)

    nhood = m.mknhood3d_aniso(1, 1.8)

    now = datetime.datetime.now()
    print(f"[{now}] Making affinity graph...")
    aff = m.seg_to_affgraph(seg, nhood)

    now = datetime.datetime.now()
    print(f"[{now}] Affinity shape: {aff.shape}")
    print(f"[{now}] Computing connected components...")
    cc, cc_sizes = m.connected_components_affgraph(aff, nhood)

    print(f"[{now}] Making affinity graph again...")
    aff2 = m.seg_to_affgraph(cc, nhood)

    print(f"[{now}] Computing connected components...")
    cc2, cc_sizes2 = m.connected_components_affgraph(aff2, nhood)

    now = datetime.datetime.now()
    print(f"[{now}] Comparing 'seg' and 'cc':")
    v_rand, v_rand_split, v_rand_merge = m.compute_V_rand_N2(seg, cc)
    print(f"[{now}]\tV_rand: {v_rand:.6f}, V_rand_split: {v_rand_split:.6f}, "
          f"V_rand_merge: {v_rand_merge:.6f}")

    now = datetime.datetime.now()
    print(f"[{now}] Comparing 'cc' and 'cc2':")
    v_rand, v_rand_split, v_rand_merge = m.compute_V_rand_N2(cc, cc2)
    print(f"[{now}]\tV_rand: {v_rand:.6f}, V_rand_split: {v_rand_split:.6f}, "
          f"V_rand_merge: {v_rand_merge:.6f}")

    return seg, cc, cc2, aff


def main():
    """Run all tests."""
    print("=" * 80)
    print("Running MALIS tests")
    print("=" * 80)

    test_neighborhood_isotropic_3d()
    print()

    test_neighborhood_anisotropic_3d()
    print()

    test_malis_loss_weights_basic()
    print()

    test_connected_components_basic()
    print()

    test_full_volume_processing()
    print()

    print("=" * 80)
    print("All tests completed")
    print("=" * 80)


if __name__ == '__main__':
    main()
