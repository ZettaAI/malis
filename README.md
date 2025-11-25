# MALIS

#### Installation
- pip install malis
- build from source: ./make.sh

#### Structured loss function for supervised learning of segmentation and clustering

Python wrapper for C++ functions for computing the MALIS loss, with **PyTorch** integration.

The MALIS loss is described here:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

#### PyTorch Usage

```python
import torch
import malis

# Create sample data
affs = torch.rand(3, 10, 10, 10)  # predicted affinities
gt_affs = torch.randint(0, 2, (3, 10, 10, 10)).float()  # ground truth affinities
gt_seg = torch.randint(0, 5, (10, 10, 10))  # ground truth segmentation
neighborhood = malis.mknhood3d(1)  # 6-connected neighborhood

# Compute MALIS loss
loss = malis.malis_loss_op(affs, gt_affs, gt_seg, neighborhood)
loss.backward()  # Compute gradients

# Or compute just the weights for custom loss functions
weights = malis.malis_weights_op(affs, gt_affs, gt_seg, neighborhood)
custom_loss = torch.sum(weights * your_loss_function(affs, gt_affs))
```

