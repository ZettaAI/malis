"""Test PyTorch integration for MALIS loss.

This script tests that the PyTorch conversion works correctly.
"""

import torch
import numpy as np
import malis


def test_pytorch_basic():
    """Test basic PyTorch functionality with MALIS loss."""
    print("Testing PyTorch integration...")

    # Create simple 3D test data (small volume)
    gt_seg = torch.tensor([
        [[0, 1, 1],
         [1, 1, 2],
         [2, 2, 2]]
    ], dtype=torch.long)

    neighborhood = malis.mknhood3d(1)  # 6-connected neighborhood
    n_edges = len(neighborhood)

    # Create affinity predictions (random)
    affs = torch.rand(n_edges, 1, 3, 3, requires_grad=True)

    # Create ground truth affinities
    gt_affs = torch.randint(0, 2, (n_edges, 1, 3, 3)).float()

    print(f"Affinity shape: {affs.shape}")
    print(f"Ground truth affinity shape: {gt_affs.shape}")
    print(f"Segmentation shape: {gt_seg.shape}")
    print(f"Number of edges: {n_edges}")

    # Test weights computation
    print("\nComputing MALIS weights...")
    weights = malis.malis_weights_op(affs, gt_affs, gt_seg, neighborhood)
    print(f"Weights shape: {weights.shape}")
    print(f"Weights type: {type(weights)}")
    print(f"Weights device: {weights.device}")

    # Test loss computation
    print("\nComputing MALIS loss...")
    loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood)
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss type: {type(loss)}")

    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    print(f"Gradient shape: {affs.grad.shape}")
    print(f"Gradient non-zero elements: {torch.count_nonzero(affs.grad).item()}")

    print("\n✓ PyTorch integration test passed!")
    return True


def test_pytorch_3d():
    """Test PyTorch with 3D data."""
    print("\n" + "="*60)
    print("Testing PyTorch with 3D data...")

    # Create 3D test data
    gt_seg = torch.randint(0, 5, (5, 5, 5), dtype=torch.long)
    neighborhood = malis.mknhood3d(1)  # 6-connected neighborhood

    # Create affinity predictions
    n_edges = len(neighborhood)
    affs = torch.rand(n_edges, 5, 5, 5, requires_grad=True)
    gt_affs = torch.randint(0, 2, (n_edges, 5, 5, 5)).float()

    print(f"3D segmentation shape: {gt_seg.shape}")
    print(f"Number of edges: {n_edges}")
    print(f"Affinities shape: {affs.shape}")

    # Compute loss
    print("\nComputing 3D MALIS loss...")
    loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood)
    print(f"Loss: {loss.item():.6f}")

    # Test backward
    print("Testing backward pass...")
    loss.backward()
    print(f"Gradient shape: {affs.grad.shape}")
    print(f"Gradient mean: {affs.grad.mean().item():.6f}")

    print("\n✓ 3D PyTorch test passed!")
    return True


def test_cuda_if_available():
    """Test CUDA support if available."""
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("CUDA not available, skipping GPU test")
        return True

    print("\n" + "="*60)
    print("Testing PyTorch with CUDA...")

    # Create test data on GPU
    device = torch.device('cuda:0')
    gt_seg = torch.randint(0, 5, (5, 5, 5), dtype=torch.long, device=device)
    neighborhood = malis.mknhood3d(1)

    n_edges = len(neighborhood)
    affs = torch.rand(n_edges, 5, 5, 5, requires_grad=True, device=device)
    gt_affs = torch.randint(0, 2, (n_edges, 5, 5, 5), device=device).float()

    print(f"Data device: {affs.device}")

    # Compute loss on GPU
    loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood)
    print(f"Loss device: {loss.device}")
    print(f"Loss: {loss.item():.6f}")

    # Test backward
    loss.backward()
    print(f"Gradient device: {affs.grad.device}")

    print("\n✓ CUDA test passed!")
    return True


if __name__ == '__main__':
    print("="*60)
    print("MALIS PyTorch Integration Tests")
    print("="*60)

    try:
        test_pytorch_basic()
        test_pytorch_3d()
        test_cuda_if_available()

        print("\n" + "="*60)
        print("All tests passed successfully!")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
