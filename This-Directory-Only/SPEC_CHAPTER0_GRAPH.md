# SPEC — Chapter 0 lesson structure and knowledge graph (web edition)

Status: DRAFT for sign-off · 2026-09-13 · author: Fable (session 50b9b490)
Companion data: `This-Directory-Only/chapter0_graph_draft.json` (machine-checked; regenerate with the
builder in the session scratchpad — it will move to `scripts/` once signed off).

## 1. Goal and the decision it drives

Give the deployed web app a complete, ordered knowledge graph for ARENA chapter 0 (0.0–0.5), so the
adaptive queue, the KG tab and the placement test can carry a learner from the existing ar-02 CNN
nodes through optimizers, a from-scratch autograd and VAEs/GANs — with **integrated nodes** that make
the learner synthesise several earlier concepts, not recall one. The decision this spec settles:
*which nodes exist, in what order, with which prerequisite and which encompassing edges*, before any
page or drill is written (each KC costs ~16 drills; the graph is the cheap part to change).

Seth's constraints (verbatim): "connect to the graph that we created … consistent with it for the
ordering … with both the encompassing edges and the prerequisite edges. the more encompassing a node
is, the more it is integrated. every encompassing edge is a prerequisite edge, but not every
prerequisite edge is an encompassing edge." Deployed edition only; Colab is abandoned.

## 2. Scope

**In:** the 0.2 gap (Sequential, ResidualBlock, BlockGroup, ResNet34, weight copy / feature
extraction, the MNIST `train` loop, hooks), all of 0.3 that runs on one CPU in the grader, all of
0.4, all of 0.5 including the ConvTranspose bonus. 42 new KCs, 18 of them integrated, 72
segments. Three new lessons `ar-03`, `ar-04`, `ar-05` (topic PyTorch, same shape as `ar-02`).

**Out (with reason):** wandb runs and sweeps (network), `broadcast`/`all_reduce`/ring all-reduce and
`DistResNetTrainer` (multi-process/GPU), `.to(device)`/pin-memory (CPU sandbox), CIFAR/MNIST/
torchvision-weight downloads (no network in the grader — drills use synthetic tensors and
mini-models), Colab notebooks. The atom graph's wandb/dist atoms stay tag-less.

## 3. Graph model

- **Prerequisite edge** `A → B` (registry `prereqs`): B is gated on A. Unchanged semantics; the
  registry order must stay a linear extension (audited).
- **Encompassing edge** `A ⇒ B` — a *subset* of B's prereqs with a weight in (0, 1]: a success on B is
  evidence on A worth `weight` of a direct success. This is the KC-level twin of the atom graph's
  `is_encompassing` / `propagation_weight`, which stays the credit-propagation authority (BKT/FIRe via
  the crosswalk). The KC-level edge is what the KG tab draws and what the audit checks.
- **Integration index** of a node = size of its transitive encompassing closure. It is what "more
  encompassing ⇒ more integrated" means operationally; it is computed, never hand-set. Top of the
  chapter: `bp.train-from-scratch` 24, `gen.gan-training-step` 22, `gen.elbo-loss` 21, `gen.reconstruction-loop` 19, `opt.finetune-loop` 17, `gen.own-layers-generator` 17.
- **Storage:** a new optional registry field on each KC, `"encompassing": {"<prereq-kc>": <weight>}`.
  `kc_graph.py` copies only the fields it knows, so the backend ignores it. New audits in
  `audit_graph_structure.check_registry`: `registry|encompassing-not-prereq`, `registry|encompassing-weight`,
  and (after authoring) `crosswalk|encompassing-unrealised` = a KC-level edge with no atom-level
  `is_encompassing` edge between the two KCs' atoms. `lesson-graph.js` draws encompassing edges solid
  and weighted, plain prereqs dashed, and shows the integration index in the node card.
- **Backfill:** the 17 existing chapter-0/einops KCs whose atoms already carry `is_encompassing`
  edges get the derived KC-level edge (table in §6) so old and new nodes obey one rule.
- **Deferred, on purpose:** per-drill encompassing credit (which prereq a *specific* drill activates)
  is still the open item from `SPEC_WORKED_EXAMPLE_SCHEDULE.md`. This spec keeps credit at the
  node level.

Ordering: the seven `ar-02` additions are spliced after `cnn.batch-normalization`; `ar-03`, `ar-04`,
`ar-05` follow in ARENA order. Every new KC sits after all its prereqs (checked).

## 4. Nodes

### 4.1 ar-02 gap (ARENA 0.2 — appended)
| # | id | title | kind | seg | prereqs | encompassing (weight) | ARENA defs | integ |
|---|---|---|---|---|---|---|---|---|
| 1 | `cnn.sequential` | Chain modules in order | concept | 2 | cnn.module-state, cnn.batch-normalization, python.control-flow, python.lists-and-tuples | cnn.module-state (0.5) | Sequential | 1 |
| 2 | `cnn.residual-block` | A residual block adds its input back | integrated | 2 | cnn.sequential, cnn.convolution-2d, cnn.batch-normalization | cnn.sequential (0.5), cnn.convolution-2d (0.4), cnn.batch-normalization (0.3) | ResidualBlock | 4 |
| 3 | `cnn.block-group` | Stack residual blocks, stride only in the first | concept | 1 | cnn.residual-block, python.control-flow | cnn.residual-block (0.7) | BlockGroup | 5 |
| 4 | `cnn.resnet34` | Assemble ResNet34 from stem, groups and head | integrated | 2 | cnn.block-group, cnn.pooling, cnn.linear-layer, cnn.sequential | cnn.block-group (0.5), cnn.pooling (0.3), cnn.sequential (0.3), cnn.linear-layer (0.2) | ResNet34 | 8 |
| 5 | `cnn.feature-extraction` | Reuse pretrained weights: copy, freeze, replace the head | concept | 2 | cnn.resnet34, cnn.module-state, cnn.linear-layer | cnn.module-state (0.3), cnn.resnet34 (0.2) | copy_weights, get_resnet_for_feature_extraction | 9 |
| 6 | `cnn.training-loop` | A training loop over batches: forward, loss, backward, step, validate | integrated | 2 | cnn.mlp, tensor.classifier-evaluation, python.control-flow, cnn.batch-normalization, cnn.feature-extraction | cnn.mlp (0.3), tensor.classifier-evaluation (0.3), cnn.feature-extraction (0.3) | SimpleMLPTrainingArgs, train | 4 |
| 7 | `cnn.forward-hooks` | Hooks watch a module's output | concept | 1 | cnn.sequential, cnn.module-state | cnn.module-state (0.3) | hook_check_for_nan_output, add_hook, remove_hooks | 1 |

### 4.2 ar-03 — ARENA 0.3 Optimization
| # | id | title | kind | seg | prereqs | encompassing (weight) | ARENA defs | integ |
|---|---|---|---|---|---|---|---|---|
| 1 | `opt.training-step` | One optimizer step: zero, backward, step | concept | 2 | cnn.training-loop, cnn.module-state, python.control-flow, numpy.stack-concat-interleave | cnn.training-loop (0.3) | opt_fn_with_sgd | 3 |
| 2 | `opt.sgd-momentum` | SGD with momentum and weight decay, in place | concept | 2 | opt.training-step, cnn.batch-normalization | opt.training-step (0.3), cnn.batch-normalization (0.2) | SGD | 5 |
| 3 | `opt.rmsprop` | RMSprop scales each step by a running RMS of the gradient | concept | 1 | opt.sgd-momentum, numpy.elementwise-ufuncs | opt.sgd-momentum (0.5) | RMSprop | 6 |
| 4 | `opt.adam` | Adam: two moments and bias correction | concept | 2 | opt.rmsprop, opt.sgd-momentum | opt.rmsprop (0.5), opt.sgd-momentum (0.3) | Adam | 7 |
| 5 | `opt.adamw` | AdamW decouples weight decay from the gradient | concept | 1 | opt.adam | opt.adam (0.7) | AdamW | 8 |
| 6 | `opt.parameter-groups` | Per-group hyperparameters | concept | 2 | opt.sgd-momentum, python.lists-and-tuples, python.control-flow | opt.sgd-momentum (0.5) | SGD (param groups) | 6 |
| 7 | `opt.finetune-loop` | A training class: step, evaluate, train | integrated | 2 | opt.adam, opt.training-step, cnn.training-loop, cnn.feature-extraction, tensor.classifier-evaluation, python.control-flow | cnn.training-loop (0.5), opt.training-step (0.2), cnn.feature-extraction (0.3), tensor.classifier-evaluation (0.3), opt.adam (0.2) | ResNetFinetuner | 17 |

### 4.3 ar-04 — ARENA 0.4 Backpropagation
| # | id | title | kind | seg | prereqs | encompassing (weight) | ARENA defs | integ |
|---|---|---|---|---|---|---|---|---|
| 1 | `bp.elementwise-backward` | A backward function turns grad_out into grad_in | concept | 2 | numpy.elementwise-ufuncs, numpy.broadcasting-rules, python.defining-functions | — | log_back, negative_back, exp_back | 0 |
| 2 | `bp.unbroadcast` | Sum a gradient back to the shape it was broadcast from | concept | 2 | numpy.broadcasting-rules, numpy.axis-reductions, python.control-flow | numpy.broadcasting-rules (0.3), numpy.axis-reductions (0.3) | unbroadcast | 2 |
| 3 | `bp.binary-backward` | Backward for two-argument ops: pick the argument, then unbroadcast | integrated | 2 | bp.elementwise-backward, bp.unbroadcast, numpy.boolean-masking | bp.unbroadcast (0.6), bp.elementwise-backward (0.4) | multiply_back0, multiply_back1, maximum_back0, maximum_back1 | 4 |
| 4 | `bp.manual-chain` | Chain backward functions by hand | integrated | 1 | bp.binary-backward, bp.elementwise-backward | bp.binary-backward (0.6), bp.elementwise-backward (0.4) | forward_and_back | 5 |
| 5 | `bp.backward-registry` | A lookup from (forward fn, argnum) to its backward | concept | 1 | python.lists-and-tuples, python.defining-functions, python.dots-and-imports | — | BackwardFuncLookup | 0 |
| 6 | `bp.recipe-and-tensor` | A Tensor remembers how it was made | concept | 2 | cnn.module-state, python.control-flow, python.lists-and-tuples | cnn.module-state (0.2) | Recipe, Tensor, log_forward, multiply_forward | 1 |
| 7 | `bp.wrap-forward` | Wrap any array function into a differentiable one | concept | 2 | bp.recipe-and-tensor, python.defining-functions | bp.recipe-and-tensor (0.6) | wrap_forward_fn, _sum, _argmax, add_, sub_ | 2 |
| 8 | `bp.topological-sort` | Order the graph so parents come first | concept | 2 | bp.recipe-and-tensor, python.control-flow, python.lists-and-tuples | bp.recipe-and-tensor (0.2) | topological_sort, get_children, sorted_computational_graph | 2 |
| 9 | `bp.backprop` | Walk the sorted graph and push gradients back | integrated | 2 | bp.topological-sort, bp.backward-registry, bp.wrap-forward, bp.binary-backward, bp.recipe-and-tensor | bp.topological-sort (0.5), bp.backward-registry (0.4), bp.binary-backward (0.3), bp.recipe-and-tensor (0.3) | backprop | 9 |
| 10 | `bp.shape-backward` | Backward for ops that move data: reshape, permute, expand, sum | concept | 2 | bp.unbroadcast, numpy.reshape-flatten, numpy.transpose-axes, numpy.sorting | bp.unbroadcast (0.3) | reshape_back, permute_back, expand_back, sum_back | 3 |
| 11 | `bp.indexing-backward` | Backward through indexing scatters the gradient | concept | 1 | bp.elementwise-backward, tensor.indexed-selection, numpy.constructors | tensor.indexed-selection (0.3) | coerce_index, _getitem, getitem_back | 1 |
| 12 | `bp.matmul-backward` | Backward for matmul: transpose the other operand | concept | 2 | bp.binary-backward, numpy.dot-matmul-patterns, numpy.transpose-axes | bp.binary-backward (0.3) | _matmul2d, matmul2d_back0, matmul2d_back1, relu | 5 |
| 13 | `bp.autograd-module` | Parameter and Module on your own Tensor | integrated | 3 | cnn.mlp, cnn.linear-layer, cnn.module-state, bp.recipe-and-tensor, bp.matmul-backward | cnn.module-state (0.4), cnn.linear-layer (0.3), cnn.mlp (0.2), bp.recipe-and-tensor (0.2) | Parameter, Module, Linear, ReLU, MLP | 4 |
| 14 | `bp.cross-entropy` | cross_entropy from a log-softmax you can differentiate | integrated | 1 | tensor.stable-probabilities, tensor.classifier-evaluation, bp.indexing-backward, bp.shape-backward | tensor.classifier-evaluation (0.3), tensor.stable-probabilities (0.3), bp.indexing-backward (0.2) | cross_entropy | 4 |
| 15 | `bp.autograd-sgd` | An SGD that updates your own Tensors, with tracking off | integrated | 2 | opt.sgd-momentum, bp.autograd-module, bp.wrap-forward | opt.sgd-momentum (0.4), bp.autograd-module (0.2), bp.wrap-forward (0.2) | NoGrad, SGD | 11 |
| 16 | `bp.train-from-scratch` | Train an MLP with nothing but your own autograd | integrated | 1 | bp.backprop, bp.autograd-sgd, bp.cross-entropy, bp.autograd-module, cnn.training-loop | bp.backprop (0.4), bp.autograd-sgd (0.3), bp.cross-entropy (0.3), bp.autograd-module (0.3), cnn.training-loop (0.3) | train, test | 24 |

### 4.4 ar-05 — ARENA 0.5 VAEs and GANs
| # | id | title | kind | seg | prereqs | encompassing (weight) | ARENA defs | integ |
|---|---|---|---|---|---|---|---|---|
| 1 | `gen.autoencoder` | Encoder to a bottleneck, decoder back | integrated | 2 | cnn.sequential, cnn.convolution-2d, cnn.linear-layer, numpy.reshape-flatten | cnn.sequential (0.4), cnn.convolution-2d (0.3), cnn.linear-layer (0.2) | Autoencoder | 4 |
| 2 | `gen.reconstruction-loop` | Train to reconstruct: MSE between output and input | integrated | 1 | gen.autoencoder, opt.finetune-loop, cnn.training-loop | cnn.training-loop (0.5), opt.finetune-loop (0.3), gen.autoencoder (0.3) | AutoencoderTrainer | 19 |
| 3 | `gen.vae-reparameterization` | Sample the latent with the reparameterisation trick | concept | 2 | gen.autoencoder, tensor.stable-probabilities, numpy.random-samplers | gen.autoencoder (0.5) | VAE | 5 |
| 4 | `gen.elbo-loss` | Reconstruction plus a KL penalty, weighted by beta | integrated | 1 | gen.vae-reparameterization, gen.reconstruction-loop, numpy.axis-reductions | gen.reconstruction-loop (0.5), gen.vae-reparameterization (0.3) | VAETrainer | 21 |
| 5 | `gen.activation-modules` | Tanh, LeakyReLU and Sigmoid as modules | concept | 1 | cnn.module-state, numpy.elementwise-ufuncs, numpy.boolean-masking | cnn.module-state (0.2) | Tanh, LeakyReLU, Sigmoid | 1 |
| 6 | `gen.dcgan-generator` | Generator: a latent vector up to an image | integrated | 2 | gen.autoencoder, cnn.sequential, cnn.batch-normalization, gen.activation-modules, cnn.block-group | gen.autoencoder (0.3), cnn.sequential (0.3), cnn.batch-normalization (0.2), cnn.block-group (0.2) | Generator | 8 |
| 7 | `gen.dcgan-discriminator` | Discriminator: an image down to one probability | integrated | 2 | gen.dcgan-generator, cnn.convolution-2d, cnn.sequential, gen.activation-modules | gen.dcgan-generator (0.3), cnn.convolution-2d (0.2), cnn.sequential (0.2) | Discriminator, DCGAN | 9 |
| 8 | `gen.weight-init` | Initialise weights the DCGAN way | concept | 1 | gen.dcgan-discriminator, cnn.module-state | cnn.module-state (0.3) | initialize_weights | 1 |
| 9 | `gen.gan-training-step` | Two optimisers, two losses, detach between them | integrated | 2 | gen.dcgan-discriminator, gen.dcgan-generator, gen.weight-init, opt.adam, cnn.training-loop, gen.reconstruction-loop, tensor.stable-probabilities | gen.reconstruction-loop (0.4), gen.dcgan-discriminator (0.3), gen.dcgan-generator (0.3), cnn.training-loop (0.3), opt.adam (0.2) | DCGANTrainer | 22 |
| 10 | `gen.conv-transpose-1d` | Transposed convolution is convolution of a padded input with the flipped kernel | concept | 3 | cnn.convolution-1d, torch.slice-assignment, numpy.constructors, numpy.transpose-axes | cnn.convolution-1d (0.6), torch.slice-assignment (0.3) | conv_transpose1d_minimal, fractional_stride_1d, conv_transpose1d | 2 |
| 11 | `gen.conv-transpose-2d` | The 2-D transposed convolution and its module | concept | 2 | gen.conv-transpose-1d, cnn.convolution-2d, cnn.module-state | gen.conv-transpose-1d (0.6), cnn.convolution-2d (0.3) | fractional_stride_2d, conv_transpose2d, ConvTranspose2d | 4 |
| 12 | `gen.own-layers-generator` | Rebuild the generator from your own ConvTranspose2d, Tanh and BatchNorm | integrated | 1 | gen.conv-transpose-2d, gen.dcgan-generator, gen.activation-modules, cnn.batch-normalization, cnn.feature-extraction | gen.conv-transpose-2d (0.4), gen.dcgan-generator (0.4), gen.activation-modules (0.3), cnn.batch-normalization (0.2), cnn.feature-extraction (0.2) | Generator (own layers, section bonus) | 17 |

### 4.5 What each node teaches (segment sketch)
- `cnn.sequential` — _modules registration via add_module, __getitem__, forward iterates in order; nn.Sequential as the torch twin
- `cnn.residual-block` — left branch conv3-bn-relu-conv3-bn; right = identity unless first_stride>1 or channels change → conv1-bn; relu(left+right). Shape rule k3 p1 stride s
- `cnn.block-group` — first block carries first_stride + in→out channel change, the rest are identity-shaped
- `cnn.resnet34` — stem conv7 s2 p3 → bn → relu → maxpool3 s2 p1; four groups [3,4,6,3] with strides [1,2,2,2]; avgpool → flatten → linear. Drills: compute output shapes per stage, build a MINI variant (2 groups, few channels) and check parameter count / output shape
- `cnn.feature-extraction` — seg1 copy_weights = zip two state_dicts by position; seg2 freeze all params, swap the final linear for n_classes. Drill checks trainable-param count
- `cnn.training-loop` — seg1 one epoch: for imgs, labels in loader: logits → cross_entropy → zero_grad/backward/step, record loss.item(); seg2 validation accuracy under no_grad + model.eval(). The 0.2 `train` exercise; deployed drills use TensorDataset on synthetic tensors, CPU only
- `cnn.forward-hooks` — register_forward_hook on every submodule via apply; remove via the handle / _forward_hooks.clear()
- `opt.training-step` — seg1 a raw leaf tensor with requires_grad → loss.backward → .grad (no module in sight); seg2 opt_fn_with_sgd: the three-line loop with torch.optim.SGD on the point itself, stacking detached copies into the trajectory
- `opt.sgd-momentum` — g = grad + wd·p; b = μ·b + g; p -= lr·b, all under no_grad/inference_mode; zero_grad sets grad None. Encompasses batch-norm's EMA-buffer + no_grad habits
- `opt.rmsprop` — v = α·v + (1-α)·g²; optional momentum buffer on g/(√v+ε)
- `opt.adam` — seg1 m,v EMAs + step counter; seg2 bias correction m/(1-β1^t), v/(1-β2^t). NOTE: encompassing opt.sgd-momentum is transitive-through-rmsprop but declared directly because the first moment IS the momentum buffer
- `opt.adamw` — p -= lr·wd·p before the moment update instead of g += wd·p. One decision, but a separately-failable one
- `opt.parameter-groups` — accept an iterable of params OR a list of dicts; fill defaults per group; raise if a param is in two groups; step loops groups
- `opt.finetune-loop` — training_step(imgs, labels) → loss; evaluate() → accuracy under no_grad with model.eval(); train() loops epochs. Deployed drills use a small MLP/CNN on synthetic tensors (no CIFAR, no wandb)
- `bp.elementwise-backward` — grad_in = grad_out · (local derivative evaluated at the input); seg1 log/negative, seg2 exp (reuse the OUTPUT), signature (grad_out, out, x)
- `bp.unbroadcast` — seg1 sum away the extra leading dims; seg2 sum keepdim over dims where the original size was 1
- `bp.binary-backward` — seg1 multiply_back0/1 (scalars allowed on either side); seg2 maximum_back with ties split ½ each way
- `bp.manual-chain` — forward d = log(log(a)·b) etc, then walk back from ones_like(d) through log_back / multiply_back0/1, accumulating; the only 'do the whole thing by hand' node
- `bp.backward-registry` — add_back_func(fn, argnum, back_fn) / get_back_func(fn, argnum); keys are (callable, int) tuples
- `bp.recipe-and-tensor` — seg1 Recipe(func,args,kwargs,parents) + Tensor(array, requires_grad, grad, recipe); seg2 forward fns: compute on .array, requires_grad = tracking_on and any(parent.requires_grad), attach recipe with parents {argnum: tensor}, leaves get NO recipe
- `bp.wrap-forward` — seg1 the wrapper: unpack Tensor args to arrays, call, build the Recipe only when is_differentiable and tracking; seg2 in-place ops (add_/sub_) and non-differentiable ones (argmax) — why in-place breaks a recorded graph
- `bp.topological-sort` — seg1 generic DFS post-order over get_children with permanent + temporary sets (temporary hit = cycle → raise); seg2 sorted_computational_graph = reversed topo over recipe.parents values
- `bp.backprop` — grads dict keyed by node; for each node in sorted order pop its outgrad; leaf & requires_grad → accumulate into .grad; else for every (argnum, parent) call lookup(recipe.func, argnum)(outgrad, node.array, *args, **kwargs) and add into grads[parent]
- `bp.shape-backward` — seg1 reshape_back (reshape to x.shape) + permute_back (inverse perm via argsort); seg2 expand_back (= unbroadcast) + sum_back (re-insert the summed dim then broadcast up)
- `bp.indexing-backward` — zeros_like(x) then scatter-ADD grad_out at the index (repeated indices accumulate); coerce Tensor indices to arrays
- `bp.matmul-backward` — seg1 back0 = grad_out @ y.T, back1 = x.T @ grad_out (shape-check them); seg2 relu = maximum(x, 0) reusing maximum_back
- `bp.autograd-module` — seg1 Parameter subclass + Module with _modules/_parameters dicts, __setattr__ routing, parameters(recurse); seg2 Linear with uniform(±1/√in) on YOUR Tensor; seg3 MLP = flatten → linear → relu → linear. Encompasses ar-02's module-state/linear/mlp because the learner re-derives them
- `bp.cross-entropy` — -log_softmax(logits)[range(n), labels] using only the wrapped ops (exp, sum, log, getitem, subtract); mean over the batch. Cross-section: 0.0 stable-probabilities + classifier-evaluation rebuilt on your autograd
- `bp.autograd-sgd` — seg1 NoGrad context manager flips the module-level grad_tracking_enabled and restores it; seg2 SGD.step under NoGrad: p.add_(-lr·p.grad); zero_grad → grad None. Cross-section 0.3 + 0.4
- `bp.train-from-scratch` — capstone: batches → MLP → cross_entropy → backward via backprop → SGD.step; test() with argmax (non-differentiable op). Deployed drill: tiny synthetic 2-D dataset, assert loss falls / accuracy > threshold in N steps
- `gen.autoencoder` — seg1 encoder conv4-s2-p1 ×2 → flatten → linear → latent; seg2 decoder mirrors it with ConvTranspose2d, output shape rule (H-1)·s - 2p + k. Drills check shapes at every stage
- `gen.reconstruction-loop` — training_step(img) → mse(model(img), img); Adam; evaluate = reconstruct held-out images
- `gen.vae-reparameterization` — seg1 encoder emits 2·latent → chunk into mu, logsigma; z = mu + exp(logsigma)·ε; seg2 forward returns (x', mu, logsigma)
- `gen.elbo-loss` — kl = ((σ² + μ² - 1)/2 - logσ).mean(); loss = recon + β·kl; training_step returns both parts
- `gen.activation-modules` — tanh from exp (overflow-safe form), leaky = where(x>0, x, slope·x) + extra_repr, sigmoid = 1/(1+e^-x)
- `gen.dcgan-generator` — seg1 linear → reshape to (c, s, s) → bn → relu with s = img_size/2^n_layers; seg2 n_layers blocks of ConvTranspose2d(k4,s2,p1)+bn+relu with channels halving, last block → Tanh. Same block-plumbing skill as cnn.block-group
- `gen.dcgan-discriminator` — seg1 mirror of the generator: Conv2d(k4,s2,p1)+LeakyReLU first (no bn), then blocks with bn; seg2 flatten → linear → Sigmoid, DCGAN holds both nets
- `gen.weight-init` — walk model.modules(); conv/convT/linear → N(0, 0.02); BatchNorm weight N(1, 0.02), bias 0
- `gen.gan-training-step` — seg1 discriminator step: D(real), D(fake.detach()), loss = -(log D(real) + log(1-D(fake))).mean(), clip, step; seg2 generator step: -log D(fake).mean() through D. Drill: run k alternating steps on a tiny pair of nets and assert which parameters changed in which step
- `gen.conv-transpose-1d` — seg1 minimal: pad k-1 both sides, flip the kernel and swap its in/out axes, conv1d; seg2 fractional stride = zeros with strided slice-assignment; seg3 stride+padding: fractional-stride then pad k-1-p, output length (L-1)·s - 2p + k
- `gen.conv-transpose-2d` — seg1 the 2-D functional with force_pair stride/padding; seg2 the module with uniform(±1/√(out·kh·kw)) weight
- `gen.own-layers-generator` — capstone: swap every torch layer for the learner's own, copy weights across (copy_weights) and assert the two generators agree on the same noise

## 5. Integration ranking (transitive encompassing closure)

| rank | kc | integ | direct encompassing |
|---|---|---|---|
| 1 | `bp.train-from-scratch` | 24 | 5 |
| 2 | `gen.gan-training-step` | 22 | 5 |
| 3 | `gen.elbo-loss` | 21 | 2 |
| 4 | `gen.reconstruction-loop` | 19 | 3 |
| 5 | `opt.finetune-loop` | 17 | 5 |
| 6 | `gen.own-layers-generator` | 17 | 5 |
| 7 | `bp.autograd-sgd` | 11 | 3 |
| 8 | `cnn.feature-extraction` | 9 | 2 |
| 9 | `bp.backprop` | 9 | 4 |
| 10 | `gen.dcgan-discriminator` | 9 | 3 |
| 11 | `cnn.resnet34` | 8 | 4 |
| 12 | `opt.adamw` | 8 | 1 |

Integrated nodes are the ones a learner cannot pass by recall: `bp.train-from-scratch` needs a working
`backprop`, a Module tree on the custom Tensor, a differentiable cross-entropy and an SGD that runs with
tracking off; `gen.gan-training-step` needs both nets, the init, Adam and the detach discipline;
`gen.own-layers-generator` swaps every torch layer for the learner's own and must match torch bit-for-bit.

## 6. Backfill of encompassing edges on existing KCs (derived from atom edges via the crosswalk)

| kc | encompassing (weight = max atom propagation_weight) |
|---|---|
| `einops.dl-flatten-heads` | einops.merge-axes (0.7), einops.split-axes (0.7) |
| `einops.split-axes` | einops.merge-axes (0.7) |
| `einops.merge-axes` | einops.pattern-language (0.7) |
| `einops.reduce-model` | einops.pattern-language (0.7) |
| `einops.repeat-model` | einops.split-axes (0.7) |
| `einops.grids-montage` | einops.split-axes (0.7) |
| `einops.pooling` | einops.split-axes (0.7), einops.reduce-model (0.7) |
| `numpy.dot-matmul-patterns` | numpy.linalg-basics (0.6) |
| `einops.channel-groups-temporal` | einops.dl-flatten-heads (0.7), einops.pooling (0.7) |
| `cnn.module-state` | numpy.dtype-astype (0.4), numpy.ndarray-model (0.4) |
| `numpy.axis-reductions` | numpy.aggregations (0.4), numpy.broadcasting-rules (0.4) |
| `tensor.stable-probabilities` | numpy.broadcasting-rules (0.4) |
| `tensor.classifier-evaluation` | tensor.stable-probabilities (0.4) |
| `cnn.convolution-2d` | cnn.module-state (0.4) |
| `cnn.linear-layer` | numpy.dot-matmul-patterns (0.4) |
| `cnn.stride-views` | numpy.dot-matmul-patterns (0.4) |
| `torch.out-argument` | numpy.ranges (0.4) |

## 7. Blind-authoring diff against the atom graph (post-hoc)

Authored from the 0.2–0.5 notebooks/solutions first, then diffed against `arena_drillable_v1.json`
(the compaction summary carried five example atom edges; noted, not used).

- **Agreed:** every 0.3–0.5 atom cluster that is drillable on CPU maps to a node above (chain-rule /
  backward-fn-signature / max-back-tied-half → `bp.elementwise-backward`/`bp.binary-backward`;
  wrap-forward / requires-grad-propagation / non-diff-fn-wrap → `bp.recipe-and-tensor`/`bp.wrap-forward`;
  dfs-three-set-toposort → `bp.topological-sort`; dispatch/pop-outgrad/accumulate-on-leaf → `bp.backprop`;
  training-step-cycle / trainer-class-skeleton → `cnn.training-loop`/`opt.finetune-loop`; ema-first/second-moment,
  bias-correction, weight-decay-decoupled → `opt.adam`/`opt.adamw`; params-iterable-vs-groups → `opt.parameter-groups`;
  reparameterization-trick / kl-closed-form / elbo → `gen.vae-*`/`gen.elbo-loss`; detach-stop-gradient /
  two-optimizers-alternating-step → `gen.gan-training-step`; convT-as-flipped-padded-conv /
  fractional-stride-zero-insertion → `gen.conv-transpose-*`; residual-skip-add ⇒ block-group-stack (0.7) adopted).
- **Changed after the diff:** added `cnn.training-loop` (the 0.2 `train` exercise and its dataloader atoms
  had no node); `bp.topological-sort` now names the three-set cycle detection; `Tensor.unbind` added to
  `opt.training-step`.
- **Disputed, kept:** the atom graph has `fractional-stride-zero-insertion ⇒ convtranspose-bn-activation-block`
  (generator encompasses transposed-conv internals). ARENA teaches the generator with `nn.ConvTranspose2d`
  as a black box and implements the layer as a bonus afterwards; the draft follows ARENA's order and
  puts that synthesis in `gen.own-layers-generator` instead.

## 8. Assumptions to confirm

1. **0.4 dialect.** ARENA 0.4 builds autograd over numpy arrays. The bank is torch-dialect and the numpy
   pages declare torch symbols, so the draft builds the custom `Tensor` over a plain `torch.Tensor`
   with autograd off (same idea, same code shape, no `np.*` symbol ratchet failures).
2. **0.4 grader support.** Drills from `bp.backprop` onward need the learner's function to run against a
   reference autograd (Recipe/Tensor/lookup/wrap). Proposal: ship `dd_autograd.py` (the ARENA solution,
   torch-backed) into the sandbox so `setup_code` is `from dd_autograd import *`; the alternative is
   ~200 lines of setup per case, which the ▶ test-check would print.
3. **Volume and phasing.** Floors alone are 522 drills (Faded 2/segment, Solo 6, Integrated 3 per KC);
   the ar-02 rate is ~16/KC → ~672. Build per lesson, each phase shipped complete (pages, drills,
   atoms+tags, glossary, caps, exercise map, notebook, guards): **A** ar-02 gap (7 KCs) + the schema/
   audit/KG rendering + backfill; **B** ar-03; **C** ar-04; **D** ar-05.
4. `cnn.forward-hooks` and `opt.adamw` are small single-segment nodes kept because each is a
   separately-failable decision (node-boundary test); drop either on request.

## 9. Sign-off checklist

- [ ] node list and titles per lesson
- [ ] prereq edges + registry order
- [ ] encompassing subsets + weights, integration ranking reads right
- [ ] storage = registry `encompassing` field + audits + KG rendering; backfill of the 17 existing KCs
- [ ] assumptions 1–3
- [ ] slice 1 = Phase A
