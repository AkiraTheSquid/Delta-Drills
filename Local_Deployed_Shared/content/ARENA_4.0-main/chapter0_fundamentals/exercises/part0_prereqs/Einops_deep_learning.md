Skip to content
Einops
Einops tutorial, part 2: deep learning
Search

 arogozhnikov/einops
v0.8.1
9.2k
385
Einops
Introduction
Tutorials
Einops tutorial, part 1: basics
Einops tutorial, part 2: deep learning
einops.pack and einops.unpack
EinMix: universal toolkit for advanced MLP architectures
Pytorch
API Reference
Testimonials
Community/Ecosystem
Table of contents
What's in this tutorial?
Select your flavour
Simple computations
Worked!
Backpropagation
Meet
Common building blocks of deep learning
Reductions
1d, 2d and 3d pooling are defined in a similar way
Good exercises
Squeeze and unsqueeze (expand_dims)
keepdims-like behavior for reductions
Stacking
Concatenation
Shuffling within a dimension
Split a dimension
Getting into the weeds of tensor packing
Shape parsing
Striding anything
Layers
What's now?
Einops tutorial, part 2: deep learning
Previous part of tutorial provides visual examples with numpy.

What's in this tutorial?¶
working with deep learning packages
important cases for deep learning models
einops.asnumpy and einops.layers
from einops import rearrange, reduce
import numpy as np

x = np.random.RandomState(42).normal(size=[10, 32, 100, 200])
# utility to hide answers
from utils import guess
Select your flavour
Switch to the framework you're most comfortable with.

# select "tensorflow" or "pytorch"
flavour = "pytorch"
print(f"selected {flavour} backend")
if flavour == "tensorflow":
    import tensorflow as tf

    tape = tf.GradientTape(persistent=True)
    tape.__enter__()
    x = tf.Variable(x) + 0
else:
    assert flavour == "pytorch"
    import torch

    x = torch.from_numpy(x)
    x.requires_grad = True
selected pytorch backend
type(x), x.shape
(torch.Tensor, torch.Size([10, 32, 100, 200]))
Simple computations
converting bchw to bhwc format and back is a common operation in CV
try to predict output shape and then check your guess!
y = rearrange(x, "b c h w -> b h w c")
guess(y.shape)
Answer is: (10, 100, 200, 32) (hover to see)
Worked!
Did you notice? Code above worked for you backend of choice.
Einops functions work with any tensor like they are native to the framework.

Backpropagation
gradients are a corner stone of deep learning
You can back-propagate through einops operations
(just as with framework native operations)
y0 = x
y1 = reduce(y0, "b c h w -> b c", "max")
y2 = rearrange(y1, "b c -> c b")
y3 = reduce(y2, "c b -> ", "sum")

if flavour == "tensorflow":
    print(reduce(tape.gradient(y3, x), "b c h w -> ", "sum"))
else:
    y3.backward()
    print(reduce(x.grad, "b c h w -> ", "sum"))
tensor(320., dtype=torch.float64)
Meet einops.asnumpy
Just converts tensors to numpy (and pulls from gpu if necessary)

from einops import asnumpy

y3_numpy = asnumpy(y3)

print(type(y3_numpy))
<class 'numpy.ndarray'>
Common building blocks of deep learning
Let's check how some familiar operations can be written with einops

Flattening is common operation, frequently appears at the boundary between convolutional layers and fully connected layers

y = rearrange(x, "b c h w -> b (c h w)")
guess(y.shape)
Answer is: (10, 640000) (hover to see)
space-to-depth

y = rearrange(x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=2, w1=2)
guess(y.shape)
Answer is: (10, 128, 50, 100) (hover to see)
depth-to-space (notice that it's reverse of the previous)

y = rearrange(x, "b (h1 w1 c) h w -> b c (h h1) (w w1)", h1=2, w1=2)
guess(y.shape)
Answer is: (10, 8, 200, 400) (hover to see)
Reductions
Simple global average pooling.

y = reduce(x, "b c h w -> b c", reduction="mean")
guess(y.shape)
Answer is: (10, 32) (hover to see)
max-pooling with a kernel 2x2

y = reduce(x, "b c (h h1) (w w1) -> b c h w", reduction="max", h1=2, w1=2)
guess(y.shape)
Answer is: (10, 32, 50, 100) (hover to see)
# you can skip names for reduced axes
y = reduce(x, "b c (h 2) (w 2) -> b c h w", reduction="max")
guess(y.shape)
Answer is: (10, 32, 50, 100) (hover to see)
1d, 2d and 3d pooling are defined in a similar way
for sequential 1-d models, you'll probably want pooling over time

reduce(x, '(t 2) b c -> t b c', reduction='max')
for volumetric models, all three dimensions are pooled

reduce(x, 'b c (x 2) (y 2) (z 2) -> b c x y z', reduction='max')
Uniformity is a strong point of einops, and you don't need specific operation for each particular case.

Good exercises
write a version of space-to-depth for 1d and 3d (2d is provided above)
write an average / max pooling for 1d models.
Squeeze and unsqueeze (expand_dims)
# models typically work only with batches,
# so to predict a single image ...
image = rearrange(x[0, :3], "c h w -> h w c")
# ... create a dummy 1-element axis ...
y = rearrange(image, "h w c -> () c h w")
# ... imagine you predicted this with a convolutional network for classification,
# we'll just flatten axes ...
predictions = rearrange(y, "b c h w -> b (c h w)")
# ... finally, decompose (remove) dummy axis
predictions = rearrange(predictions, "() classes -> classes")
keepdims-like behavior for reductions
empty composition () provides dimensions of length 1, which are broadcastable.
alternatively, you can use just 1 to introduce new axis, that's a synonym to ()
per-channel mean-normalization for each image:

y = x - reduce(x, "b c h w -> b c 1 1", "mean")
guess(y.shape)
Answer is: (10, 32, 100, 200) (hover to see)
per-channel mean-normalization for whole batch:

y = x - reduce(y, "b c h w -> 1 c 1 1", "mean")
guess(y.shape)
Answer is: (10, 32, 100, 200) (hover to see)
Stacking
let's take a list of tensors

list_of_tensors = list(x)
New axis (one that enumerates tensors) appears first on the left side of expression. Just as if you were indexing list - first you'd get tensor by index

tensors = rearrange(list_of_tensors, "b c h w -> b h w c")
guess(tensors.shape)
Answer is: (10, 100, 200, 32) (hover to see)
# or maybe stack along last dimension?
tensors = rearrange(list_of_tensors, "b c h w -> h w c b")
guess(tensors.shape)
Answer is: (100, 200, 32, 10) (hover to see)
Concatenation
concatenate over the first dimension?

tensors = rearrange(list_of_tensors, "b c h w -> (b h) w c")
guess(tensors.shape)
Answer is: (1000, 200, 32) (hover to see)
or maybe concatenate along last dimension?

tensors = rearrange(list_of_tensors, "b c h w -> h w (b c)")
guess(tensors.shape)
Answer is: (100, 200, 320) (hover to see)
Shuffling within a dimension
channel shuffle (as it is drawn in shufflenet paper)

y = rearrange(x, "b (g1 g2 c) h w-> b (g2 g1 c) h w", g1=4, g2=4)
guess(y.shape)
Answer is: (10, 32, 100, 200) (hover to see)
simpler version of channel shuffle

y = rearrange(x, "b (g c) h w-> b (c g) h w", g=4)
guess(y.shape)
Answer is: (10, 32, 100, 200) (hover to see)
Split a dimension
Here's a super-convenient trick.

Example: when a network predicts several bboxes for each position

Assume we got 8 bboxes, 4 coordinates each.
To get coordinated into 4 separate variables, you move corresponding dimension to front and unpack tuple.

bbox_x, bbox_y, bbox_w, bbox_h = rearrange(x, "b (coord bbox) h w -> coord b bbox h w", coord=4, bbox=8)
# now you can operate on individual variables
max_bbox_area = reduce(bbox_w * bbox_h, "b bbox h w -> b h w", "max")
guess(bbox_x.shape)
guess(max_bbox_area.shape)
Answer is: (10, 8, 100, 200) (hover to see)
Answer is: (10, 100, 200) (hover to see)
Getting into the weeds of tensor packing
you can skip this part - it explains why taking a habit of defining splits and packs explicitly

when implementing custom gated activation (like GLU), split is needed:

y1, y2 = rearrange(x, 'b (split c) h w -> split b c h w', split=2)
result = y2 * sigmoid(y2) # or tanh
... but we could split differently

y1, y2 = rearrange(x, 'b (c split) h w -> split b c h w', split=2)
first one splits channels into consequent groups: y1 = x[:, :x.shape[1] // 2, :, :]
while second takes channels with a step: y1 = x[:, 0::2, :, :]
This may drive to very surprising results when input is

a result of group convolution
a result of bidirectional LSTM/RNN
multi-head attention
Let's focus on the second case (LSTM/RNN), since it is less obvious.

For instance, cudnn concatenates LSTM outputs for forward-in-time and backward-in-time

Also in pytorch GLU splits channels into consequent groups (first way) So when LSTM's output comes to GLU,

forward-in-time produces linear part, and backward-in-time produces activation ...
and role of directions is different, and gradients coming to two parts are different
that's not what you expect from simple GLU(BLSTM(x)), right?
einops notation makes such inconsistencies explicit and easy-detectable

Shape parsing
just a handy utility

from einops import parse_shape
def convolve_2d(x):
    # imagine we have a simple 2d convolution with padding,
    # so output has same shape as input.
    # Sorry for laziness, use imagination!
    return x
# imagine we are working with 3d data
x_5d = rearrange(x, "b c x (y z) -> b c x y z", z=20)
# but we have only 2d convolutions.
# That's not a problem, since we can apply
y = rearrange(x_5d, "b c x y z -> (b z) c x y")
y = convolve_2d(y)
# not just specifies additional information, but verifies that all dimensions match
y = rearrange(y, "(b z) c x y -> b c x y z", **parse_shape(x_5d, "b c x y z"))
parse_shape(x_5d, "b c x y z")
{'b': 10, 'c': 32, 'x': 100, 'y': 10, 'z': 20}
# we can skip some dimensions by writing underscore
parse_shape(x_5d, "batch c _ _ _")
{'batch': 10, 'c': 32}
Striding anything
Finally, how to convert any operation into a strided operation?
(like convolution with strides, aka dilated/atrous convolution)

# each image is split into subgrids, each subgrid now is a separate "image"
y = rearrange(x, "b c (h hs) (w ws) -> (hs ws b) c h w", hs=2, ws=2)
y = convolve_2d(y)
# pack subgrids back to an image
y = rearrange(y, "(hs ws b) c h w -> b c (h hs) (w ws)", hs=2, ws=2)

assert y.shape == x.shape
Layers
For frameworks that prefer operating with layers, layers are available.

You'll need to import a proper one depending on your backend:

from einops.layers.torch import Rearrange, Reduce
from einops.layers.flax import Rearrange, Reduce
from einops.layers.tensorflow import Rearrange, Reduce
from einops.layers.chainer import Rearrange, Reduce
Einops layers are identical to operations, and have same parameters.
(for the exception of first argument, which should be passed during call)

layer = Rearrange(pattern, **axes_lengths)
layer = Reduce(pattern, reduction, **axes_lengths)

# apply layer to tensor
x = layer(x)
Usually it is more convenient to use layers, not operations, to build models

# example given for pytorch, but code in other frameworks is almost identical
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Reduce

model = Sequential(
    Conv2d(3, 6, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Conv2d(6, 16, kernel_size=5),
    # combined pooling and flattening in a single step
    Reduce('b c (h 2) (w 2) -> b (c h w)', 'max'), 
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 10), 
    # In flax, the {'axis': value} syntax for specifying values for axes is mandatory:
    # Rearrange('(b1 b2) d -> b1 b2 d', {'b1': 12}), 
)
What's now?
rush through writing better code with einops+pytorch
Use different framework? Not a big issue, most recommendations transfer well to other frameworks.
einops works the same way in any framework.

Finally - just write your code with einops!

Made with Material for MkDocs
