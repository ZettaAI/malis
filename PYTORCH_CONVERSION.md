# PyTorch Conversion Summary

This document summarizes the conversion of the MALIS package from TensorFlow to PyTorch.

## Branch
All changes are on the `pytorch-conversion` branch.

## Changes Made

### 1. Core Conversion (`malis/malis_loss.py`)
- **Replaced TensorFlow imports** with PyTorch (`import torch`)
- **Converted `MalisWeights.get_edge_weights()`** to accept both PyTorch tensors and numpy arrays
  - Added automatic conversion from torch.Tensor to numpy for C++ backend compatibility
  - Returns numpy arrays that are converted back to torch tensors in the wrapper

- **Created `MalisWeightsFunction`**: Custom PyTorch autograd Function
  - Implements `forward()` for weight computation
  - Implements `backward()` for gradient propagation
  - Properly handles optional mask parameters
  - Maintains device placement (CPU/CUDA)

- **Converted `malis_weights_op()`**:
  - Changed from TensorFlow `tf.py_function` to PyTorch custom autograd function
  - Returns PyTorch tensors instead of TensorFlow tensors
  - Maintains same API signature for backward compatibility

- **Converted `malis_loss_op()`**:
  - Replaced `tf.square()` with `torch.square()`
  - Replaced `tf.reduce_sum()` with `torch.sum()`
  - Replaced `tf.multiply()` with native PyTorch `*` operator
  - Returns scalar PyTorch tensor suitable for `loss.backward()`

### 2. Dependencies

#### `setup.py`
- Added `torch>=1.9.0` to `install_requires`
- Removed TensorFlow dependency

#### `requirements.txt`
- Added `torch>=1.9.0`
- Removed TensorFlow dependency

### 3. Documentation

#### `README.md`
- Updated description to mention PyTorch integration
- Added PyTorch usage example showing:
  - Basic loss computation
  - Gradient computation with `loss.backward()`
  - Custom loss function usage with `malis_weights_op()`

### 4. Testing

#### `test_pytorch.py` (new file)
Created comprehensive test suite for PyTorch integration:
- **Basic 1D test**: Tests fundamental PyTorch operations
- **3D volumetric test**: Tests realistic 3D segmentation scenarios
- **CUDA test**: Tests GPU compatibility (if available)
- Validates gradient computation through backward pass
- Verifies device placement (CPU/CUDA)

## Key Features Maintained

✓ **Same API**: All function signatures remain the same
✓ **Backward compatibility**: Existing code using the package will work with minimal changes
✓ **C++ backend**: Core MALIS algorithm still uses optimized C++ implementation
✓ **Autograd support**: Full PyTorch gradient computation support
✓ **Device agnostic**: Works on both CPU and CUDA
✓ **Masking support**: Maintains all optional masking parameters

## Usage Changes

### Before (TensorFlow)
```python
import tensorflow as tf
import malis

affs = tf.random.uniform([3, 10, 10, 10])
gt_affs = tf.round(tf.random.uniform([3, 10, 10, 10]))
gt_seg = tf.random.uniform([10, 10, 10], maxval=5, dtype=tf.int32)
neighborhood = malis.mknhood3d(1)

loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood)
```

### After (PyTorch)
```python
import torch
import malis

affs = torch.rand(3, 10, 10, 10, requires_grad=True)
gt_affs = torch.randint(0, 2, (3, 10, 10, 10)).float()
gt_seg = torch.randint(0, 5, (10, 10, 10))
neighborhood = malis.mknhood3d(1)

loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood)
loss.backward()  # Compute gradients
```

## Testing

Run the PyTorch tests:
```bash
python test_pytorch.py
```

Run original tests (C++ backend tests):
```bash
python -m malis.test_malis
```

## Architecture

```
User Code (PyTorch tensors)
         ↓
malis_loss_op() / malis_weights_op()
         ↓
MalisWeightsFunction.apply()
         ↓
MalisWeights.get_edge_weights()
         ↓
[Tensor → NumPy conversion]
         ↓
C++ Backend (malis_loss_weights)
         ↓
[NumPy → Tensor conversion]
         ↓
PyTorch Tensor (with gradients)
```

## Performance Considerations

- **C++ backend**: Still uses optimized C++ code for core computation
- **Memory**: Temporary numpy conversion required for C++ backend
- **GPU**: Tensors are transferred to CPU for C++ computation, then back to original device
- **Gradients**: Proper gradient flow maintained through autograd system

## Next Steps

1. Test the conversion with existing workflows
2. Consider adding optional CUDA kernels for GPU acceleration
3. Add more comprehensive test cases
4. Update any dependent packages
