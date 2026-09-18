"""Build + validate the chapter-0 KC graph draft.

Authored blind from the ARENA 0.2–0.5 notebooks/solutions (per
docs/method-anchoring-controls-for-kc-authoring.md); the atom graph is only
loaded at the END for the diff check.

Writes This-Directory-Only/chapter0_graph_draft.json and prints:
  - structural checks (DAG, linear extension, encompassing ⊆ prereqs, ids known)
  - integration index per node (= |transitive encompassing closure|)
  - backfill of encompassing edges for EXISTING ar-* KCs, derived from atom edges
  - diff of the new nodes vs the 0.3–0.5 atom sub-graph
"""
import json, sys, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = json.load(open(ROOT / "Local_Deployed_Shared/lessons/kc_registry.json"))
OUT = ROOT / "This-Directory-Only/chapter0_graph_draft.json"

L = lambda i, title: {"id": i, "topic": "PyTorch", "title": title, "subtopic_key": f"PyTorch: {i}"}
LESSONS = [
    L("ar-03", "ARENA 0.3 — Optimization"),
    L("ar-04", "ARENA 0.4 — Backpropagation"),
    L("ar-05", "ARENA 0.5 — VAEs and GANs"),
]

def K(id, lesson, title, prereqs, enc=None, defs=(), sym=(), seg=2, kind="concept", note=""):
    return {"id": id, "lesson": lesson, "title": title, "prereqs": prereqs,
            "encompassing": enc or {}, "arena_defs": list(defs), "new_syntax": list(sym),
            "segments": seg, "kind": kind, "note": note}

KCS = [
    # ── 0.2 gap (appended to ar-02 after cnn.batch-normalization) ─────────────
    K("cnn.sequential", "ar-02", "Chain modules in order",
      ["cnn.module-state", "cnn.batch-normalization", "python.control-flow", "python.lists-and-tuples"],
      {"cnn.module-state": 0.5},
      defs=["Sequential"], sym=["torch.nn.Sequential", "Module.add_module", "Module._modules"], seg=2,
      note="_modules registration via add_module, __getitem__, forward iterates in order; nn.Sequential as the torch twin"),
    K("cnn.residual-block", "ar-02", "A residual block adds its input back",
      ["cnn.sequential", "cnn.convolution-2d", "cnn.batch-normalization"],
      {"cnn.sequential": 0.5, "cnn.convolution-2d": 0.4, "cnn.batch-normalization": 0.3},
      defs=["ResidualBlock"], sym=["torch.nn.Conv2d", "torch.nn.BatchNorm2d", "torch.nn.ReLU"], seg=2, kind="integrated",
      note="left branch conv3-bn-relu-conv3-bn; right = identity unless first_stride>1 or channels change → conv1-bn; relu(left+right). Shape rule k3 p1 stride s"),
    K("cnn.block-group", "ar-02", "Stack residual blocks, stride only in the first",
      ["cnn.residual-block", "python.control-flow"],
      {"cnn.residual-block": 0.7},
      defs=["BlockGroup"], seg=1,
      note="first block carries first_stride + in→out channel change, the rest are identity-shaped"),
    K("cnn.resnet34", "ar-02", "Assemble ResNet34 from stem, groups and head",
      ["cnn.block-group", "cnn.pooling", "cnn.linear-layer", "cnn.sequential"],
      {"cnn.block-group": 0.5, "cnn.pooling": 0.3, "cnn.sequential": 0.3, "cnn.linear-layer": 0.2},
      defs=["ResNet34"], sym=["torch.nn.AdaptiveAvgPool2d", "torch.nn.Flatten", "torch.nn.MaxPool2d"], seg=2, kind="integrated",
      note="stem conv7 s2 p3 → bn → relu → maxpool3 s2 p1; four groups [3,4,6,3] with strides [1,2,2,2]; avgpool → flatten → linear. Drills: compute output shapes per stage, build a MINI variant (2 groups, few channels) and check parameter count / output shape"),
    K("cnn.feature-extraction", "ar-02", "Reuse pretrained weights: copy, freeze, replace the head",
      ["cnn.resnet34", "cnn.module-state", "cnn.linear-layer"],
      {"cnn.module-state": 0.3, "cnn.resnet34": 0.2},
      defs=["copy_weights", "get_resnet_for_feature_extraction"],
      sym=["Module.state_dict", "Module.load_state_dict", "Tensor.requires_grad_", "Module.named_parameters"], seg=2,
      note="seg1 copy_weights = zip two state_dicts by position; seg2 freeze all params, swap the final linear for n_classes. Drill checks trainable-param count"),
    K("cnn.training-loop", "ar-02", "A training loop over batches: forward, loss, backward, step, validate",
      ["cnn.mlp", "tensor.classifier-evaluation", "python.control-flow", "cnn.batch-normalization", "cnn.feature-extraction"],
      {"cnn.mlp": 0.3, "tensor.classifier-evaluation": 0.3, "cnn.feature-extraction": 0.3},
      defs=["SimpleMLPTrainingArgs", "train"],
      sym=["torch.utils.data.TensorDataset", "torch.utils.data.DataLoader", "torch.optim.Adam (black box)", "Optimizer.zero_grad", "Optimizer.step", "Tensor.backward", "Tensor.item", "Module.train", "Module.eval"], seg=2, kind="integrated",
      note="seg1 one epoch: for imgs, labels in loader: logits → cross_entropy → zero_grad/backward/step, record loss.item(); seg2 validation accuracy under no_grad + model.eval(). The 0.2 `train` exercise; deployed drills use TensorDataset on synthetic tensors, CPU only"),
    K("cnn.forward-hooks", "ar-02", "Hooks watch a module's output",
      ["cnn.sequential", "cnn.module-state"],
      {"cnn.module-state": 0.3},
      defs=["hook_check_for_nan_output", "add_hook", "remove_hooks"],
      sym=["Module.register_forward_hook", "Module.apply", "torch.isnan"], seg=1,
      note="register_forward_hook on every submodule via apply; remove via the handle / _forward_hooks.clear()"),

    # ── ar-03 Optimization ────────────────────────────────────────────────────
    K("opt.training-step", "ar-03", "One optimizer step: zero, backward, step",
      ["cnn.training-loop", "cnn.module-state", "python.control-flow", "torch.stack-concat-interleave"],
      {"cnn.training-loop": 0.3},
      defs=["opt_fn_with_sgd"],
      sym=["Tensor.requires_grad", "Tensor.grad", "torch.optim.SGD", "Tensor.detach", "Tensor.unbind"], seg=2,
      note="seg1 a raw leaf tensor with requires_grad → loss.backward → .grad (no module in sight); seg2 opt_fn_with_sgd: the three-line loop with torch.optim.SGD on the point itself, stacking detached copies into the trajectory"),
    K("opt.sgd-momentum", "ar-03", "SGD with momentum and weight decay, in place",
      ["opt.training-step", "cnn.batch-normalization"],
      {"opt.training-step": 0.3, "cnn.batch-normalization": 0.2},
      defs=["SGD"], sym=["torch.inference_mode"], seg=2,
      note="g = grad + wd·p; b = μ·b + g; p -= lr·b, all under no_grad/inference_mode; zero_grad sets grad None. Encompasses batch-norm's EMA-buffer + no_grad habits"),
    K("opt.rmsprop", "ar-03", "RMSprop scales each step by a running RMS of the gradient",
      ["opt.sgd-momentum", "torch.elementwise-ops"],
      {"opt.sgd-momentum": 0.5},
      defs=["RMSprop"], seg=1,
      note="v = α·v + (1-α)·g²; optional momentum buffer on g/(√v+ε)"),
    K("opt.adam", "ar-03", "Adam: two moments and bias correction",
      ["opt.rmsprop", "opt.sgd-momentum"],
      {"opt.rmsprop": 0.5, "opt.sgd-momentum": 0.3},
      defs=["Adam"], seg=2,
      note="seg1 m,v EMAs + step counter; seg2 bias correction m/(1-β1^t), v/(1-β2^t). NOTE: encompassing opt.sgd-momentum is transitive-through-rmsprop but declared directly because the first moment IS the momentum buffer"),
    K("opt.adamw", "ar-03", "AdamW decouples weight decay from the gradient",
      ["opt.adam"],
      {"opt.adam": 0.7},
      defs=["AdamW"], seg=1,
      note="p -= lr·wd·p before the moment update instead of g += wd·p. One decision, but a separately-failable one"),
    K("opt.parameter-groups", "ar-03", "Per-group hyperparameters",
      ["opt.sgd-momentum", "python.lists-and-tuples", "python.control-flow"],
      {"opt.sgd-momentum": 0.5},
      defs=["SGD (param groups)"], sym=["dict.setdefault", "builtin.isinstance"], seg=2,
      note="accept an iterable of params OR a list of dicts; fill defaults per group; raise if a param is in two groups; step loops groups"),
    K("opt.finetune-loop", "ar-03", "A training class: step, evaluate, train",
      ["opt.adam", "opt.training-step", "cnn.training-loop", "cnn.feature-extraction", "tensor.classifier-evaluation", "python.control-flow"],
      {"cnn.training-loop": 0.5, "opt.training-step": 0.2, "cnn.feature-extraction": 0.3, "tensor.classifier-evaluation": 0.3, "opt.adam": 0.2},
      defs=["ResNetFinetuner"], sym=["torch.optim.Adam", "torch.nn.functional.cross_entropy", "Module.train", "Module.eval"], seg=2, kind="integrated",
      note="training_step(imgs, labels) → loss; evaluate() → accuracy under no_grad with model.eval(); train() loops epochs. Deployed drills use a small MLP/CNN on synthetic tensors (no CIFAR, no wandb)"),

    # ── ar-04 Backprop (torch-dialect: backing arrays are plain tensors, autograd OFF) ──
    K("bp.elementwise-backward", "ar-04", "A backward function turns grad_out into grad_in",
      ["torch.elementwise-ops", "torch.broadcasting-rules", "python.defining-functions"],
      {},
      defs=["log_back", "negative_back", "exp_back"], seg=2,
      note="grad_in = grad_out · (local derivative evaluated at the input); seg1 log/negative, seg2 exp (reuse the OUTPUT), signature (grad_out, out, x)"),
    K("bp.unbroadcast", "ar-04", "Sum a gradient back to the shape it was broadcast from",
      ["torch.broadcasting-rules", "torch.axis-reductions", "python.control-flow"],
      {"torch.broadcasting-rules": 0.3, "torch.axis-reductions": 0.3},
      defs=["unbroadcast"], seg=2,
      note="seg1 sum away the extra leading dims; seg2 sum keepdim over dims where the original size was 1"),
    K("bp.binary-backward", "ar-04", "Backward for two-argument ops: pick the argument, then unbroadcast",
      ["bp.elementwise-backward", "bp.unbroadcast", "torch.boolean-masking"],
      {"bp.unbroadcast": 0.6, "bp.elementwise-backward": 0.4},
      defs=["multiply_back0", "multiply_back1", "maximum_back0", "maximum_back1"], seg=2, kind="integrated",
      note="seg1 multiply_back0/1 (scalars allowed on either side); seg2 maximum_back with ties split ½ each way"),
    K("bp.manual-chain", "ar-04", "Chain backward functions by hand",
      ["bp.binary-backward", "bp.elementwise-backward"],
      {"bp.binary-backward": 0.6, "bp.elementwise-backward": 0.4},
      defs=["forward_and_back"], seg=1, kind="integrated",
      note="forward d = log(log(a)·b) etc, then walk back from ones_like(d) through log_back / multiply_back0/1, accumulating; the only 'do the whole thing by hand' node"),
    K("bp.backward-registry", "ar-04", "A lookup from (forward fn, argnum) to its backward",
      ["python.lists-and-tuples", "python.defining-functions", "python.dots-and-imports"],
      {},
      defs=["BackwardFuncLookup"], sym=["syntax.class", "dict (tuple keys)"], seg=1,
      note="add_back_func(fn, argnum, back_fn) / get_back_func(fn, argnum); keys are (callable, int) tuples"),
    K("bp.recipe-and-tensor", "ar-04", "A Tensor remembers how it was made",
      ["cnn.module-state", "python.control-flow", "python.lists-and-tuples"],
      {"cnn.module-state": 0.2},
      defs=["Recipe", "Tensor", "log_forward", "multiply_forward"], sym=["dataclasses.dataclass", "builtin.any"], seg=2,
      note="seg1 Recipe(func,args,kwargs,parents) + Tensor(array, requires_grad, grad, recipe); seg2 forward fns: compute on .array, requires_grad = tracking_on and any(parent.requires_grad), attach recipe with parents {argnum: tensor}, leaves get NO recipe"),
    K("bp.wrap-forward", "ar-04", "Wrap any array function into a differentiable one",
      ["bp.recipe-and-tensor", "python.defining-functions"],
      {"bp.recipe-and-tensor": 0.6},
      defs=["wrap_forward_fn", "_sum", "_argmax", "add_", "sub_"], sym=["syntax.args-kwargs", "syntax.closure"], seg=2,
      note="seg1 the wrapper: unpack Tensor args to arrays, call, build the Recipe only when is_differentiable and tracking; seg2 in-place ops (add_/sub_) and non-differentiable ones (argmax) — why in-place breaks a recorded graph"),
    K("bp.topological-sort", "ar-04", "Order the graph so parents come first",
      ["bp.recipe-and-tensor", "python.control-flow", "python.lists-and-tuples"],
      {"bp.recipe-and-tensor": 0.2},
      defs=["topological_sort", "get_children", "sorted_computational_graph"], sym=["syntax.recursion", "set"], seg=2,
      note="seg1 generic DFS post-order over get_children with permanent + temporary sets (temporary hit = cycle → raise); seg2 sorted_computational_graph = reversed topo over recipe.parents values"),
    K("bp.backprop", "ar-04", "Walk the sorted graph and push gradients back",
      ["bp.topological-sort", "bp.backward-registry", "bp.wrap-forward", "bp.binary-backward", "bp.recipe-and-tensor"],
      {"bp.topological-sort": 0.5, "bp.backward-registry": 0.4, "bp.binary-backward": 0.3, "bp.recipe-and-tensor": 0.3},
      defs=["backprop"], seg=2, kind="integrated",
      note="grads dict keyed by node; for each node in sorted order pop its outgrad; leaf & requires_grad → accumulate into .grad; else for every (argnum, parent) call lookup(recipe.func, argnum)(outgrad, node.array, *args, **kwargs) and add into grads[parent]"),
    K("bp.shape-backward", "ar-04", "Backward for ops that move data: reshape, permute, expand, sum",
      ["bp.unbroadcast", "torch.reshape-flatten", "torch.transpose-axes", "torch.sorting"],
      {"bp.unbroadcast": 0.3},
      defs=["reshape_back", "permute_back", "expand_back", "sum_back"], sym=["torch.argsort (inverse permutation)", "Tensor.expand"], seg=2,
      note="seg1 reshape_back (reshape to x.shape) + permute_back (inverse perm via argsort); seg2 expand_back (= unbroadcast) + sum_back (re-insert the summed dim then broadcast up)"),
    K("bp.indexing-backward", "ar-04", "Backward through indexing scatters the gradient",
      ["bp.elementwise-backward", "tensor.indexed-selection", "torch.constructors"],
      {"tensor.indexed-selection": 0.3},
      defs=["coerce_index", "_getitem", "getitem_back"], sym=["Tensor.index_put_", "torch.zeros_like"], seg=1,
      note="zeros_like(x) then scatter-ADD grad_out at the index (repeated indices accumulate); coerce Tensor indices to arrays"),
    K("bp.matmul-backward", "ar-04", "Backward for matmul: transpose the other operand",
      ["bp.binary-backward", "torch.dot-matmul-patterns", "torch.transpose-axes"],
      {"bp.binary-backward": 0.3},
      defs=["_matmul2d", "matmul2d_back0", "matmul2d_back1", "relu"], seg=2,
      note="seg1 back0 = grad_out @ y.T, back1 = x.T @ grad_out (shape-check them); seg2 relu = maximum(x, 0) reusing maximum_back"),
    K("bp.autograd-module", "ar-04", "Parameter and Module on your own Tensor",
      ["cnn.mlp", "cnn.linear-layer", "cnn.module-state", "bp.recipe-and-tensor", "bp.matmul-backward"],
      {"cnn.module-state": 0.4, "cnn.linear-layer": 0.3, "cnn.mlp": 0.2, "bp.recipe-and-tensor": 0.2},
      defs=["Parameter", "Module", "Linear", "ReLU", "MLP"], sym=["object.__setattr__", "Module.__call__"], seg=3, kind="integrated",
      note="seg1 Parameter subclass + Module with _modules/_parameters dicts, __setattr__ routing, parameters(recurse); seg2 Linear with uniform(±1/√in) on YOUR Tensor; seg3 MLP = flatten → linear → relu → linear. Encompasses ar-02's module-state/linear/mlp because the learner re-derives them"),
    K("bp.cross-entropy", "ar-04", "cross_entropy from a log-softmax you can differentiate",
      ["tensor.stable-probabilities", "tensor.classifier-evaluation", "bp.indexing-backward", "bp.shape-backward"],
      {"tensor.classifier-evaluation": 0.3, "tensor.stable-probabilities": 0.3, "bp.indexing-backward": 0.2},
      defs=["cross_entropy"], seg=1, kind="integrated",
      note="-log_softmax(logits)[range(n), labels] using only the wrapped ops (exp, sum, log, getitem, subtract); mean over the batch. Cross-section: 0.0 stable-probabilities + classifier-evaluation rebuilt on your autograd"),
    K("bp.autograd-sgd", "ar-04", "An SGD that updates your own Tensors, with tracking off",
      ["opt.sgd-momentum", "bp.autograd-module", "bp.wrap-forward"],
      {"opt.sgd-momentum": 0.4, "bp.autograd-module": 0.2, "bp.wrap-forward": 0.2},
      defs=["NoGrad", "SGD"], sym=["syntax.with", "object.__enter__", "object.__exit__", "syntax.global"], seg=2, kind="integrated",
      note="seg1 NoGrad context manager flips the module-level grad_tracking_enabled and restores it; seg2 SGD.step under NoGrad: p.add_(-lr·p.grad); zero_grad → grad None. Cross-section 0.3 + 0.4"),
    K("bp.train-from-scratch", "ar-04", "Train an MLP with nothing but your own autograd",
      ["bp.backprop", "bp.autograd-sgd", "bp.cross-entropy", "bp.autograd-module", "cnn.training-loop"],
      {"bp.backprop": 0.4, "bp.autograd-sgd": 0.3, "bp.cross-entropy": 0.3, "bp.autograd-module": 0.3, "cnn.training-loop": 0.3},
      defs=["train", "test"], seg=1, kind="integrated",
      note="capstone: batches → MLP → cross_entropy → backward via backprop → SGD.step; test() with argmax (non-differentiable op). Deployed drill: tiny synthetic 2-D dataset, assert loss falls / accuracy > threshold in N steps"),

    # ── ar-05 VAEs & GANs ─────────────────────────────────────────────────────
    K("gen.autoencoder", "ar-05", "Encoder to a bottleneck, decoder back",
      ["cnn.sequential", "cnn.convolution-2d", "cnn.linear-layer", "torch.reshape-flatten"],
      {"cnn.sequential": 0.4, "cnn.convolution-2d": 0.3, "cnn.linear-layer": 0.2},
      defs=["Autoencoder"], sym=["torch.nn.ConvTranspose2d (black box, shape rule)", "einops.layers.torch.Rearrange"], seg=2, kind="integrated",
      note="seg1 encoder conv4-s2-p1 ×2 → flatten → linear → latent; seg2 decoder mirrors it with ConvTranspose2d, output shape rule (H-1)·s - 2p + k. Drills check shapes at every stage"),
    K("gen.reconstruction-loop", "ar-05", "Train to reconstruct: MSE between output and input",
      ["gen.autoencoder", "opt.finetune-loop", "cnn.training-loop"],
      {"cnn.training-loop": 0.5, "opt.finetune-loop": 0.3, "gen.autoencoder": 0.3},
      defs=["AutoencoderTrainer"], sym=["torch.nn.functional.mse_loss"], seg=1, kind="integrated",
      note="training_step(img) → mse(model(img), img); Adam; evaluate = reconstruct held-out images"),
    K("gen.vae-reparameterization", "ar-05", "Sample the latent with the reparameterisation trick",
      ["gen.autoencoder", "tensor.stable-probabilities", "torch.random-samplers"],
      {"gen.autoencoder": 0.5},
      defs=["VAE"], sym=["Tensor.chunk", "torch.randn_like"], seg=2,
      note="seg1 encoder emits 2·latent → chunk into mu, logsigma; z = mu + exp(logsigma)·ε; seg2 forward returns (x', mu, logsigma)"),
    K("gen.elbo-loss", "ar-05", "Reconstruction plus a KL penalty, weighted by beta",
      ["gen.vae-reparameterization", "gen.reconstruction-loop", "torch.axis-reductions"],
      {"gen.reconstruction-loop": 0.5, "gen.vae-reparameterization": 0.3},
      defs=["VAETrainer"], seg=1, kind="integrated",
      note="kl = ((σ² + μ² - 1)/2 - logσ).mean(); loss = recon + β·kl; training_step returns both parts"),
    K("gen.activation-modules", "ar-05", "Tanh, LeakyReLU and Sigmoid as modules",
      ["cnn.module-state", "torch.elementwise-ops", "torch.boolean-masking"],
      {"cnn.module-state": 0.2},
      defs=["Tanh", "LeakyReLU", "Sigmoid"], sym=["torch.where", "Module.extra_repr"], seg=1,
      note="tanh from exp (overflow-safe form), leaky = where(x>0, x, slope·x) + extra_repr, sigmoid = 1/(1+e^-x)"),
    K("gen.dcgan-generator", "ar-05", "Generator: a latent vector up to an image",
      ["gen.autoencoder", "cnn.sequential", "cnn.batch-normalization", "gen.activation-modules", "cnn.block-group"],
      {"gen.autoencoder": 0.3, "cnn.sequential": 0.3, "cnn.batch-normalization": 0.2, "cnn.block-group": 0.2},
      defs=["Generator"], seg=2, kind="integrated",
      note="seg1 linear → reshape to (c, s, s) → bn → relu with s = img_size/2^n_layers; seg2 n_layers blocks of ConvTranspose2d(k4,s2,p1)+bn+relu with channels halving, last block → Tanh. Same block-plumbing skill as cnn.block-group"),
    K("gen.dcgan-discriminator", "ar-05", "Discriminator: an image down to one probability",
      ["gen.dcgan-generator", "cnn.convolution-2d", "cnn.sequential", "gen.activation-modules"],
      {"gen.dcgan-generator": 0.3, "cnn.convolution-2d": 0.2, "cnn.sequential": 0.2},
      defs=["Discriminator", "DCGAN"], seg=2, kind="integrated",
      note="seg1 mirror of the generator: Conv2d(k4,s2,p1)+LeakyReLU first (no bn), then blocks with bn; seg2 flatten → linear → Sigmoid, DCGAN holds both nets"),
    K("gen.weight-init", "ar-05", "Initialise weights the DCGAN way",
      ["gen.dcgan-discriminator", "cnn.module-state"],
      {"cnn.module-state": 0.3},
      defs=["initialize_weights"], sym=["Module.modules", "torch.nn.init.normal_", "torch.nn.init.constant_"], seg=1,
      note="walk model.modules(); conv/convT/linear → N(0, 0.02); BatchNorm weight N(1, 0.02), bias 0"),
    K("gen.gan-training-step", "ar-05", "Two optimisers, two losses, detach between them",
      ["gen.dcgan-discriminator", "gen.dcgan-generator", "gen.weight-init", "opt.adam", "cnn.training-loop", "gen.reconstruction-loop", "tensor.stable-probabilities"],
      {"gen.reconstruction-loop": 0.4, "gen.dcgan-discriminator": 0.3, "gen.dcgan-generator": 0.3, "cnn.training-loop": 0.3, "opt.adam": 0.2},
      defs=["DCGANTrainer"], sym=["torch.nn.utils.clip_grad_norm_", "Optimizer betas"], seg=2, kind="integrated",
      note="seg1 discriminator step: D(real), D(fake.detach()), loss = -(log D(real) + log(1-D(fake))).mean(), clip, step; seg2 generator step: -log D(fake).mean() through D. Drill: run k alternating steps on a tiny pair of nets and assert which parameters changed in which step"),
    K("gen.conv-transpose-1d", "ar-05", "Transposed convolution is convolution of a padded input with the flipped kernel",
      ["cnn.convolution-1d", "torch.slice-assignment", "torch.constructors", "torch.transpose-axes"],
      {"cnn.convolution-1d": 0.6, "torch.slice-assignment": 0.3},
      defs=["conv_transpose1d_minimal", "fractional_stride_1d", "conv_transpose1d"], sym=["Tensor.flip"], seg=3,
      note="seg1 minimal: pad k-1 both sides, flip the kernel and swap its in/out axes, conv1d; seg2 fractional stride = zeros with strided slice-assignment; seg3 stride+padding: fractional-stride then pad k-1-p, output length (L-1)·s - 2p + k"),
    K("gen.conv-transpose-2d", "ar-05", "The 2-D transposed convolution and its module",
      ["gen.conv-transpose-1d", "cnn.convolution-2d", "cnn.module-state"],
      {"gen.conv-transpose-1d": 0.6, "cnn.convolution-2d": 0.3},
      defs=["fractional_stride_2d", "conv_transpose2d", "ConvTranspose2d"], seg=2,
      note="seg1 the 2-D functional with force_pair stride/padding; seg2 the module with uniform(±1/√(out·kh·kw)) weight"),
    K("gen.own-layers-generator", "ar-05", "Rebuild the generator from your own ConvTranspose2d, Tanh and BatchNorm",
      ["gen.conv-transpose-2d", "gen.dcgan-generator", "gen.activation-modules", "cnn.batch-normalization", "cnn.feature-extraction"],
      {"gen.conv-transpose-2d": 0.4, "gen.dcgan-generator": 0.4, "gen.activation-modules": 0.3, "cnn.batch-normalization": 0.2, "cnn.feature-extraction": 0.2},
      defs=["Generator (own layers, section bonus)"], seg=1, kind="integrated",
      note="capstone: swap every torch layer for the learner's own, copy weights across (copy_weights) and assert the two generators agree on the same noise"),
]

# ── existing registry ────────────────────────────────────────────────────────
existing = {k["id"]: k for k in REG["kcs"]}
existing_order = [k["id"] for k in REG["kcs"]]
lesson_ids = [l["id"] for l in REG["lessons"]]
new_ids = [k["id"] for k in KCS]
allk = dict(existing); allk.update({k["id"]: k for k in KCS})

def fail(msg): print("FAIL", msg); FAILS.append(msg)
FAILS = []

# ids — a draft node that is already in the registry has LANDED (the ar-02 gap
# shipped 2026-09-14); the registry row must then match the draft's edges.
LANDED = [k["id"] for k in KCS if k["id"] in existing]
for k in KCS:
    if k["id"] in existing:
        row = existing[k["id"]] if isinstance(existing, dict) else None
        if row is not None:
            if set(row.get("prereqs", [])) != set(k["prereqs"]):
                fail(f"{k['id']} landed with prereqs {sorted(row.get('prereqs', []))} != draft {sorted(k['prereqs'])}")
            if dict(row.get("encompassing", {})) != dict(k["encompassing"]):
                fail(f"{k['id']} landed with encompassing {row.get('encompassing', {})} != draft {k['encompassing']}")
        continue
    for p in k["prereqs"]:
        if p not in allk: fail(f"{k['id']} unknown prereq {p}")
    for e, w in k["encompassing"].items():
        if e not in k["prereqs"]: fail(f"{k['id']} encompassing {e} is not a prereq (encompassing ⊆ prereqs)")
        if not (0 < w <= 1): fail(f"{k['id']} weight {e}={w}")
    if k["lesson"] not in lesson_ids + [l["id"] for l in LESSONS]: fail(f"{k['id']} lesson {k['lesson']}")
    if len(k["prereqs"]) != len(set(k["prereqs"])): fail(f"{k['id']} duplicate prereq")

# proposed registry order: existing, then new per lesson in listed order, ar-02 gap spliced after cnn.batch-normalization
order = []
for i in existing_order:
    order.append(i)
    if i == "cnn.batch-normalization":
        order += [k["id"] for k in KCS if k["lesson"] == "ar-02" and k["id"] not in existing_order]
order += [k["id"] for k in KCS if k["lesson"] != "ar-02" and k["id"] not in existing_order]
if len(order) != len(set(order)): fail(f"order has duplicates: {[i for i in set(order) if order.count(i) > 1]}")
pos = {i: n for n, i in enumerate(order)}
for k in KCS:
    for p in k["prereqs"]:
        if pos[p] >= pos[k["id"]]: fail(f"order: {k['id']} before its prereq {p}")

# DAG
def parents(i): return allk[i]["prereqs"]
seen = {}
def dfs(i, stack):
    if i in stack: fail("cycle " + " -> ".join(stack + [i])); return
    if seen.get(i): return
    for p in parents(i): dfs(p, stack + [i])
    seen[i] = True
for i in new_ids: dfs(i, [])

# integration index = |transitive encompassing closure|; direct enc count
enc = {k["id"]: k["encompassing"] for k in KCS}
def enc_closure(i, acc=None):
    acc = set() if acc is None else acc
    for e in enc.get(i, {}):
        if e not in acc:
            acc.add(e); enc_closure(e, acc)
    return acc

print("\n== structure:", "OK" if not FAILS else f"{len(FAILS)} problems")
print(f"{'kc':30s} {'lesson':6s} {'kind':10s} pre enc closure")
rows = []
for k in KCS:
    c = enc_closure(k["id"])
    rows.append((len(c), k["id"]))
    print(f"{k['id']:30s} {k['lesson']:6s} {k['kind']:10s} {len(k['prereqs']):3d} {len(k['encompassing']):3d} {len(c):3d}")
print("\nmost integrated:", sorted(rows, reverse=True)[:8])
kinds = collections.Counter(k["kind"] for k in KCS)
print("counts:", len(KCS), dict(kinds), collections.Counter(k["lesson"] for k in KCS), "segments", sum(k["segments"] for k in KCS))

# ── backfill: derive KC-level encompassing for EXISTING ar-*/einsum KCs from the atom graph ──
GP = ROOT / "This-Directory-Only/backend/app/data/concept_graphs/arena_drillable_v1.json"
CW = ROOT / "Local_Deployed_Shared/concept-graph/kc_atom_crosswalk.json"
g = json.load(open(GP)); cw = json.load(open(CW))
atom_kcs = collections.defaultdict(set)   # atom -> {kc}
cw_map = cw.get("kc_to_atoms") or cw.get("kcs") or cw
for kc, atoms in (cw_map.items() if isinstance(cw_map, dict) else []):
    lst = atoms if isinstance(atoms, list) else atoms.get("atoms", [])
    for a in lst:
        atom_kcs[a if isinstance(a, str) else a.get("a")].add(kc)
backfill = collections.defaultdict(dict)
for e in g["prerequisite_edges"]:
    if not e.get("is_encompassing"): continue
    for a in atom_kcs.get(e["prerequisite_id"], ()):
        for b in atom_kcs.get(e["dependent_id"], ()):
            if a == b: continue
            if b in existing and a in existing[b]["prereqs"]:
                w = e.get("propagation_weight") or e.get("weight") or 0.5
                backfill[b][a] = max(backfill[b].get(a, 0), round(w, 2))
print("\n== backfill (existing KCs; KC-level encompassing derived from atom is_encompassing edges via crosswalk)")
for b, d in sorted(backfill.items(), key=lambda x: existing_order.index(x[0])):
    print(f"  {b:32s} {d}")
print("  crosswalk keys sample:", list(cw)[:5] if isinstance(cw, dict) else type(cw))

# ── diff: atoms of ARENA 0.2–0.5 not owned by any KC (coverage gaps my design must absorb) ──
print("== atom graph keys:", list(g))
nodes = {n["id"]: n for n in next((g[k] for k in ("nodes", "atoms", "concepts") if isinstance(g.get(k), list)), [])}
print("\n== atom fields:", sorted(next(iter(nodes.values())).keys()) if nodes else None)
sec_key = next((k for k in ("section", "arena_section", "exercise", "source") if nodes and k in next(iter(nodes.values()))), None)
unowned = [n for n in nodes.values() if n["id"] not in atom_kcs]
by_sec = collections.defaultdict(list)
for n in unowned: by_sec[str(n.get(sec_key, "?"))[:12]].append(n["id"])
print("== unowned atoms by", sec_key, ":", {k: len(v) for k, v in sorted(by_sec.items())})
for k, v in sorted(by_sec.items()):
    print(f"  [{k}] " + ", ".join(sorted(v)))
enc_edges_unowned = [(e["prerequisite_id"], e["dependent_id"], e.get("propagation_weight")) for e in g["prerequisite_edges"]
                     if e.get("is_encompassing") and (e["prerequisite_id"] not in atom_kcs or e["dependent_id"] not in atom_kcs)]
print("== is_encompassing atom edges touching unowned atoms:", len(enc_edges_unowned))
for a, b, w in enc_edges_unowned: print(f"   {a} -> {b} ({w})")

draft = {
    "_comment": "Chapter-0 KC graph draft. encompassing ⊆ prereqs; weight = share of a success on this KC that counts as evidence on the prereq. integration = |transitive encompassing closure|. Order = proposed registry order (linear extension). See SPEC_CHAPTER0_GRAPH.md.",
    "lessons": LESSONS,
    "registry_order": order,
    "kcs": [dict(k, integration=len(enc_closure(k["id"]))) for k in KCS],
    "backfill_encompassing": {b: d for b, d in backfill.items()},
}
OUT.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n")
print("\nwrote", OUT, "fails:", len(FAILS))
sys.exit(1 if FAILS else 0)
