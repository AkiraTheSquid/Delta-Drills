#!/usr/bin/env python3
"""
Generate and verify alternative einops solutions for questions in the question bank.
Every generated solution is verified against the question's actual test_cases.
Outputs verified solutions to Local_Deployed_Shared/practice/einops_solutions.json.
"""

import os
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED_DIR = HERE.parent
QUESTIONS_PATH = SHARED_DIR / "questions.json"
OUT_PATH = SHARED_DIR / "practice" / "einops_solutions.json"
PROV_PATH = SHARED_DIR / "practice" / "einops_provenance.json"

# Add backend to path
BACKEND_DIR = Path("/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/backend")
if BACKEND_DIR.exists():
    sys.path.insert(0, str(BACKEND_DIR))
    from app import code_runner
    if not code_runner.preload_torch():
        raise RuntimeError(
            "PyTorch unavailable. Re-run with the backend virtualenv: "
            "/home/stellar-thread/Applications/Delta-Drills-Local/This-Directory-Only/backend/.venv/bin/python"
        )
else:
    raise RuntimeError("Backend directory not found")

with open(QUESTIONS_PATH, encoding="utf-8") as f:
    questions = json.load(f)

q_by_id = {q["id"]: q for q in questions}

CANDIDATES = {}

# =====================================================================
# Subtopic: BatchNorm per-channel broadcasting (3 solutions)
# =====================================================================
# Q459: BatchNorm's per-channel vectors (weight, bias, running mean, running v
CANDIDATES[459] = '# Solution created by Antigravity\nimport einops\nimport torch\n\ndef solve(x, weight, bias, mean, var):\n    """BatchNorm per-channel broadcasting via rearrange."""\n    w = einops.rearrange(weight, "c -> 1 c 1 1")\n    b = einops.rearrange(bias, "c -> 1 c 1 1")\n    m = einops.rearrange(mean, "c -> 1 c 1 1")\n    v = einops.rearrange(var, "c -> 1 c 1 1")\n    return (x - m) / torch.sqrt(v) * w + b\n'

# Q460: A length-C vector broadcasts against a (B, C, H, W) tensor only if its
CANDIDATES[460] = '# Solution created by Antigravity\nimport einops\nimport torch\n\ndef solve(weight):\n    """Align length-C vector to dim 1 of a 4D tensor."""\n    return einops.rearrange(weight, "c -> 1 c 1 1")\n'

# Q461: Broadcasting aligns trailing dims — so adding a raw length-C bias to a
CANDIDATES[461] = '# Solution created by Antigravity\nimport einops\nimport torch\n\ndef solve(x, bias):\n    """Bias broadcasting alignment via rearrange."""\n    C = x.shape[1]\n    b = einops.rearrange(bias, "c -> 1 c 1 1")\n    try:\n        bad = x + bias\n        return ("no error", x + b)\n    except RuntimeError:\n        return ("raised", x + b)\n'

# =====================================================================
# Subtopic: BatchNorm variance: biased vs unbiased (1 solutions)
# =====================================================================
# Q454: BatchNorm2d computes PER-CHANNEL statistics over the batch and spatial
CANDIDATES[454] = '# Solution created by Antigravity\nimport einops\nimport torch\n\ndef solve(x):\n    """Per-channel statistics over batch and spatial dims via einops."""\n    mean = einops.reduce(x, "b c h w -> c", "mean")\n    diff = x - einops.rearrange(mean, "c -> 1 c 1 1")\n    var = einops.reduce(diff ** 2, "b c h w -> c", "mean")\n    return (mean, var)\n'

# =====================================================================
# Subtopic: Conv2d module mechanics (1 solutions)
# =====================================================================
# Q415: Write a function solve(x, w) that computes a VALID 1-D convolution (cr
CANDIDATES[415] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, w):\n    unfolded = x.unfold(2, w.shape[2], 1)\n    return einops.einsum(unfolded, w, "b i l k, o i k -> b o l")\n'

# =====================================================================
# Subtopic: Core array literacy (16 solutions)
# =====================================================================
# Q46: Write a function solve(n) that takes a positive integer n and returns 
CANDIDATES[46] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(n):\n    return einops.rearrange(t.arange(n * n), "(h w) -> h w", h=n, w=n)\n'

# Q490: Write a function solve(x) that takes a PyTorch tensor of any shape and
CANDIDATES[490] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    """Return x collapsed to a single axis."""\n    return einops.rearrange(x, \'... -> (...)\')\n'

# Q491: Write a function solve(x, cols) that takes a tensor and a column count
CANDIDATES[491] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, cols):\n    flat = einops.rearrange(x, "... -> (...)")\n    return einops.rearrange(flat, "(r c) -> r c", c=cols)\n'

# Q492: Write a function solve(n, rows) that returns a 2-D tensor holding the 
CANDIDATES[492] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(n, rows):\n    return einops.rearrange(t.arange(n), "(r c) -> r c", r=rows)\n'

# Q495: Write a function solve(x) that takes a PyTorch tensor and returns the 
CANDIDATES[495] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.reduce(x, "... -> 1", "sum").item()\n'

# Q496: Write a function solve(x) that returns a tuple (mean, count): the mean
CANDIDATES[496] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    mean_val = einops.reduce(x, "... -> 1", "mean").item()\n    return (mean_val, x.numel())\n'

# Q558: Write a function solve(rows) that builds a tensor a and returns the TR
CANDIDATES[558] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(rows):\n    """Return a.T\'s contents as a plain nested list."""\n    a = t.tensor(rows)\n    return einops.rearrange(a, \'i j -> j i\').tolist()\n'

# Q610: Write a function solve(rows) that builds a tensor from the nested list
CANDIDATES[610] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows):\n    a = t.tensor(rows)\n    return tuple(einops.rearrange(a, "i j -> j i").shape)\n'

# Q611: Write a function solve(rows) that builds a tensor from the nested list
CANDIDATES[611] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows):\n    a = t.tensor(rows)\n    return einops.rearrange(a, "i j -> j i").tolist()[0]\n'

# Q612: Write a function solve(rows) that builds a tensor a and returns a tupl
CANDIDATES[612] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows):\n    a = t.tensor(rows)\n    back = einops.rearrange(einops.rearrange(a, "i j -> j i"), "j i -> i j")\n    return (t.equal(back, a), tuple(back.shape))\n'

# Q613: Write a function solve(rows) that builds a tensor a and returns a tupl
CANDIDATES[613] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows):\n    a = t.tensor(rows)\n    view = einops.rearrange(a, "i j -> j i")\n    return (view.tolist(), tuple(view.shape))\n'

# Q614: Write a function solve(rows) that builds a tensor a and returns a tupl
CANDIDATES[614] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows):\n    a = t.tensor(rows)\n    view = einops.rearrange(a, "i j -> j i")\n    return (tuple(a.shape), tuple(view.shape), view.tolist(), a.ndim)\n'

# Q1315: Given a nonempty float tensor x of any shape and a Python float thresh
CANDIDATES[1315] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, threshold):\n    frac = einops.reduce((x > threshold).float(), "... -> ", "mean")\n    return float(frac)\n'

# Q1325: Given nonempty float tensors x and weights with identical shapes, retu
CANDIDATES[1325] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, weights):\n    prod_sum = einops.reduce(x * weights, "... -> ", "sum")\n    w_sum = einops.reduce(weights, "... -> ", "sum")\n    return float(prod_sum / w_sum)\n'

# Q1329: Given nonempty float tensors x and target with identical shapes, retur
CANDIDATES[1329] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, target):\n    num = einops.reduce(x * target, "... -> ", "sum")\n    den = einops.reduce(x * x, "... -> ", "sum")\n    return float(num / den)\n'

# Q1331: Given nonempty float tensors x and weights with identical shapes, retu
CANDIDATES[1331] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, weights):\n    total = einops.reduce(weights, "... -> ", "sum")\n    mean = einops.reduce(x * weights, "... -> ", "sum") / total\n    return float(einops.reduce(weights * (x - mean) ** 2, "... -> ", "sum") / total)\n'

# =====================================================================
# Subtopic: Matrix multiplication backward (dL/dX = dL/dM @ Y^T, dL/dY = X^T @ dL/dM) (1 solutions)
# =====================================================================
# Q451: Apply the matmul backward rules directly — no autograd. For M = XY: dL
CANDIDATES[451] = '# Solution created by Antigravity\nimport einops\nimport torch\n\ndef solve(X, Y, dLdM):\n    """Matmul backward rules via einsum: dL/dX = (dL/dM) Y^T, dL/dY = X^T (dL/dM)."""\n    dLdX = einops.einsum(dLdM, Y, "n m, k m -> n k")\n    dLdY = einops.einsum(X, dLdM, "n k, n m -> k m")\n    return (dLdX, dLdY)\n'

# =====================================================================
# Subtopic: Pooling, Flatten, BatchNorm (2 solutions)
# =====================================================================
# Q418: Global average pooling collapses each channel's spatial map to a singl
CANDIDATES[418] = "# Solution created by Antigravity\nimport torch\nimport einops\n\ndef solve(x):\n    return einops.reduce(x, 'b c h w -> b c', 'mean')\n"

# Q419: Between a conv stack and a linear head, activations of shape (B, C, H,
CANDIDATES[419] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.rearrange(x, "b ... -> b (...)")\n'

# =====================================================================
# Subtopic: Rays as tensors (6 solutions)
# =====================================================================
# Q821: Given a ray `[[ox, oy, oz], [dx, dy, dz]]` and a list of parameters `u
CANDIDATES[821] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ray, us):\n    """Many points on one ray, by broadcasting u down a column."""\n    ray = t.tensor(ray)\n    u = t.tensor(us)\n    u_col = einops.rearrange(u, "k -> k 1")\n    return (ray[0] + u_col * ray[1]).tolist()\n'

# Q826: Return the points at parameters `0, 1, …, n-1` on the ray `[[ox, oy, o
CANDIDATES[826] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ray, n):\n    """n integer steps along one ray."""\n    ray = t.tensor(ray)\n    u = einops.rearrange(t.arange(n), "n -> n 1")\n    return (ray[0] + u * ray[1]).tolist()\n'

# Q827: Given a stack of rays as a nested list of shape (n, 2, 3) and one scal
CANDIDATES[827] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rays, u):\n    """One u, many rays."""\n    rays = t.tensor(rays)\n    origins = einops.rearrange(rays[:, 0], "n d -> n d")\n    dirs = einops.rearrange(rays[:, 1], "n d -> n d")\n    return (origins + u * dirs).tolist()\n'

# Q828: Given a stack of rays (n, 2, 3) as a nested list and a target x-coordi
CANDIDATES[828] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rays, x):\n    """Per-ray u from x, then per-ray y, vectorised."""\n    rays = t.tensor(rays)\n    origins = rays[:, 0]\n    dirs = rays[:, 1]\n    u = (x - origins[:, 0]) / dirs[:, 0]\n    u_col = einops.rearrange(u, "n -> n 1")\n    return (origins[:, 1] + u * dirs[:, 1]).tolist()\n'

# Q829: Given a ray `[[ox, oy, oz], [dx, dy, dz]]` and a list of x-coordinates
CANDIDATES[829] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ray, xs):\n    """Many x\'s -> many u\'s -> many points, one broadcast."""\n    ray = t.tensor(ray)\n    u = (t.tensor(xs) - ray[0, 0]) / ray[1, 0]\n    u_col = einops.rearrange(u, "k -> k 1")\n    return (ray[0] + u_col * ray[1]).tolist()\n'

# Q839: A 2-D fan: `(ny * nz, 2, 3)` rays with direction `[1, y, z]` for every
CANDIDATES[839] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ny, nz, y_limit, z_limit):\n    """make_rays_2d: a grid of directions in one canvas."""\n    rays = t.zeros((ny, nz, 2, 3))\n    rays[:, :, 1, 0] = 1\n    y_vals = t.linspace(-y_limit, y_limit, ny)\n    z_vals = t.linspace(-z_limit, z_limit, nz)\n    rays[:, :, 1, 1] = einops.repeat(y_vals, "ny -> ny nz", nz=nz)\n    rays[:, :, 1, 2] = einops.repeat(z_vals, "nz -> ny nz", ny=ny)\n    return einops.rearrange(rays, "ny nz r d -> (ny nz) r d").tolist()\n'

# =====================================================================
# Subtopic: Vectorization and broadcasting (56 solutions)
# =====================================================================
# Q23: Write a function solve(z) that takes a 2-D PyTorch tensor and returns 
CANDIDATES[23] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(z):\n    return einops.rearrange(z, 'h w -> (w h)')\n"

# Q37: Write a function solve(a, b) that takes two 1-D PyTorch tensors of flo
CANDIDATES[37] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    return float(einops.einsum(a, b, "i, i -> "))\n'

# Q62: Write a function solve(z) that takes a 1-D PyTorch integer tensor and 
CANDIDATES[62] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(z):\n    return int(einops.reduce(z, "i -> ", "sum"))\n'

# Q84: Write a function solve(a, b) that takes two PyTorch float tensors of i
CANDIDATES[84] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(a, b):\n    return einops.reduce([a, b], 'two ... -> ...', 'mean')\n"

# Q89: Write a function solve(a, b) that takes two 1-D PyTorch tensors of the
CANDIDATES[89] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    return einops.rearrange(t.stack((a, b), dim=1), "n two -> (n two)")\n'

# Q95: Write a function solve(img) that takes an RGB image as a 3-D PyTorch t
CANDIDATES[95] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(img):\n    return einops.einsum(img, t.tensor([0.299, 0.587, 0.114]), '... c, c -> ...')\n"

# Q121: Write a function solve(a, b) that takes two 2-D float tensors of ident
CANDIDATES[121] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    return einops.einsum(a, b, "i j, i j -> i")\n'

# Q135: Write a function solve(x) that takes a 4-D PyTorch integer tensor x of
CANDIDATES[135] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    return einops.reduce(x, '... h w -> ...', 'sum')\n"

# Q141: Write a function solve(a, b) that takes two square float matrices of t
CANDIDATES[141] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    return einops.einsum(a, b, "i j, j i -> i")\n'

# Q144: Write a function solve(z, v) that takes a 2-D float tensor z of shape 
CANDIDATES[144] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(z, v):\n    return einops.einsum(z, v, 'n m, m -> n')\n"

# Q151: Write a function solve(a, b) that takes a 3-D float tensor a of shape 
CANDIDATES[151] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(a, b):\n    return a * einops.rearrange(b, 'h w -> h w 1')\n"

# Q220: Write a function solve(x) that takes a 2-D PyTorch integer matrix x an
CANDIDATES[220] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    return einops.reduce(x, 'h w -> w', 'sum')\n"

# Q501: Write a function solve(x, scale) where x has shape (rows, cols) and sc
CANDIDATES[501] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, scale):\n    """Scale every row by its own factor."""\n    return x * einops.rearrange(scale, "r -> r 1")\n'

# Q502: Write a function solve(rows, cols) returning a (rows, cols) integer te
CANDIDATES[502] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows, cols):\n    """Build multiplication table i * j via einsum outer product."""\n    return einops.einsum(t.arange(rows), t.arange(cols), "r, c -> r c")\n'

# Q503: Write a function solve(x) that takes a 2-D integer tensor and returns 
CANDIDATES[503] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    """Return the sum of each ROW."""\n    return einops.reduce(x, \'h w -> h\', \'sum\')\n'

# Q504: Write a function solve(x) that returns the per-row sums of a 2-D tenso
CANDIDATES[504] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    """Return the row sums with the reduced axis KEPT."""\n    return einops.reduce(x, \'h w -> h 1\', \'sum\')\n'

# Q505: Write a function solve(x) that takes a 2-D float tensor with no zero r
CANDIDATES[505] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    """Make every row of a float tensor sum to 1."""\n    return x / einops.reduce(x, \'h w -> h 1\', \'sum\')\n'

# Q509: Write a function solve(a, b) that takes two square float matrices and 
CANDIDATES[509] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    """Elementwise product and matrix product."""\n    return (a * b, einops.einsum(a, b, "i k, k j -> i j"))\n'

# Q510: Write a function solve(a) that returns a tuple (values, shape) for the
CANDIDATES[510] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(a):\n    """Return the transpose of a 2-D tensor, and its shape."""\n    tr = einops.rearrange(a, \'i j -> j i\')\n    return (tr.tolist(), tuple(tr.shape))\n'

# Q511: Write a function solve(a, x) where a has shape (m, n) and x has shape 
CANDIDATES[511] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, x):\n    """Return a @ x for a matrix and a vector, plus the result shape."""\n    r = einops.einsum(a, x, "m n, n -> m")\n    return (r.tolist(), tuple(r.shape))\n'

# Q513: Write a function solve(mats, x) where mats has shape (batch, m, n) and
CANDIDATES[513] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(mats, x):\n    """Apply a BATCH of matrices to one vector."""\n    r = einops.einsum(mats, x, "b m n, n -> b m")\n    return (r.tolist(), tuple(r.shape))\n'

# Q514: Write a function solve(a, b) that takes two 1-D float tensors of the s
CANDIDATES[514] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    """Dot product via einsum contraction."""\n    return float(einops.einsum(a, b, "i, i -> "))\n'

# Q515: Write a function solve(x) that takes a 2-D float tensor with no all-ze
CANDIDATES[515] = '# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    """Scale every row to unit length."""\n    return x / einops.reduce(x.pow(2), \'h w -> h 1\', \'sum\').sqrt()\n'

# Q948: Two 1-D PyTorch tensors are given: `a` of length m and `b` of length n
CANDIDATES[948] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    return einops.rearrange(a, "m -> m 1") - einops.rearrange(b, "n -> 1 n")\n'

# Q950: `x` has shape (b, n, d) — a batch of b feature matrices — and `s` has 
CANDIDATES[950] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, s):\n    """Scale each sample in batch by s."""\n    return x * einops.rearrange(s, "b -> b 1 1")\n'

# Q951: Return the (rows, cols) integer tensor whose entry at row i, column j 
CANDIDATES[951] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(rows, cols):\n    return einops.rearrange(t.arange(rows * cols), "(r c) -> r c", c=cols)\n'

# Q952: Two 1-D PyTorch tensors are given: `a` of length m and `b` of length n
CANDIDATES[952] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    """(m, n) table where entry [i, j] = a[i] * b[j]."""\n    return einops.einsum(a, b, "m, n -> m n")\n'

# Q953: `imgs` has shape (b, h, w, c) — a batch of images with the colour axis
CANDIDATES[953] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(imgs, mean):\n    """Each channel of every image centered by channel mean."""\n    return imgs - einops.rearrange(mean, "c -> 1 1 1 c")\n'

# Q954: `x` has shape (rows, cols), `row_scale` has shape (rows,) and `col_bia
CANDIDATES[954] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, row_scale, col_bias):\n    """Rows scaled by row_scale, then cols shifted by col_bias."""\n    return x * einops.rearrange(row_scale, "r -> r 1") + einops.rearrange(col_bias, "c -> 1 c")\n'

# Q955: Three 1-D PyTorch tensors are given: `a` of length i, `b` of length j 
CANDIDATES[955] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b, c):\n    return einops.einsum(a, b, c, "i, j, k -> i j k")\n'

# Q956: `imgs` has shape (b, h, w, c) with the colour axis last, `mean` has sh
CANDIDATES[956] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(imgs, mean, gain):\n    centered = imgs - einops.rearrange(mean, "c -> 1 1 1 c")\n    return centered * einops.rearrange(gain, "b -> b 1 1 1")\n'

# Q958: `origins` and `directions` both have shape (b, 3) — b rays in 3-D — an
CANDIDATES[958] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(origins, directions, ts):\n    """(b, k, 3) points along b rays at k parameters."""\n    o = einops.rearrange(origins, "b d -> b 1 d")\n    d = einops.rearrange(directions, "b d -> b 1 d")\n    t_expanded = einops.rearrange(ts, "k -> 1 k 1")\n    return o + t_expanded * d\n'

# Q974: Two 2-D tensors with the same number of columns are given. Return the 
CANDIDATES[974] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    """Matrices joined top to bottom."""\n    out, _ = einops.pack([a, b], "* c")\n    return out\n'

# Q975: Two 2-D tensors with the same number of ROWS are given. Return the sin
CANDIDATES[975] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    """Matrices joined side by side."""\n    out, _ = einops.pack([a, b], "r *")\n    return out\n'

# Q976: Two 1-D tensors of the same length are given. Return the 1-D tensor th
CANDIDATES[976] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    """Two vectors alternated."""\n    return einops.rearrange(t.stack((a, b), dim=1), "n two -> (n two)")\n'

# Q978: A Python list of tensors, all the same shape, is given. Return the ten
CANDIDATES[978] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(mats):\n    return einops.reduce(t.stack(mats), "k ... -> ...", "max")\n'

# Q979: Three 1-D integer tensors of the same length n are given. Return the 1
CANDIDATES[979] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b, c):\n    """Three vectors interleaved."""\n    return einops.rearrange(t.stack((a, b, c), dim=1), "n three -> (n three)")\n'

# Q981: Two 2-D tensors of the SAME shape are given. Return a tuple of two tup
CANDIDATES[981] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(a, b):\n    v, _ = einops.pack([a, b], "* c")\n    h, _ = einops.pack([a, b], "r *")\n    return ((v, tuple(v.shape)), (h, tuple(h.shape)))\n'

# Q983: A Python list of tensors, all the same shape, is given. Return a tuple
CANDIDATES[983] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(mats):\n    """The pile, its elementwise mean, and how many were piled."""\n    piled = t.stack(mats, dim=0)\n    mean = einops.reduce(piled, "k ... -> ...", "mean")\n    return piled, mean, int(piled.shape[0])\n'

# Q1363: Given a 3-D float tensor `x` of shape `(n, r, c)`, return a 2-D tensor
CANDIDATES[1363] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Row totals for every one of the n matrices."""\n    return einops.reduce(x, "n r c -> n r", "sum")\n'

# Q1364: Given a 3-D float tensor `x` of shape `(n, r, c)` — n matrices, one be
CANDIDATES[1364] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Average the n matrices down to one."""\n    return einops.reduce(x, "n r c -> r c", "mean")\n'

# Q1365: Given a 2-D float tensor `x` of shape `(r, c)`, return the mean of eac
CANDIDATES[1365] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Column means that still have two axes."""\n    return einops.reduce(x, "r c -> 1 c", "mean")\n'

# Q1366: Given a 3-D float tensor `x` of shape `(n, h, w)`, return a tensor of 
CANDIDATES[1366] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """One mean per slice, shaped to broadcast back over it."""\n    return einops.reduce(x, "n h w -> n 1 1", "mean")\n'

# Q1367: Given a 2-D float tensor `x` of shape `(r, c)`, return a 1-D tensor of
CANDIDATES[1367] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Each column\'s mean, relative to the grand mean."""\n    col_mean = einops.reduce(x, "r c -> c", "mean")\n    grand_mean = einops.reduce(x, "r c -> ", "mean")\n    return col_mean - grand_mean\n'

# Q1368: Given a 3-D float tensor `imgs` of shape `(b, h, w)` — a batch of b gr
CANDIDATES[1368] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(imgs):\n    """One number per image: its average brightness."""\n    return einops.reduce(imgs, "b h w -> b", "mean")\n'

# Q1369: Given a 2-D float tensor `x` of shape `(r, c)`, return a tensor of the
CANDIDATES[1369] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Centre each row on its own mean."""\n    return x - einops.reduce(x, "r c -> r 1", "mean")\n'

# Q1370: Given a 2-D float tensor `x` of shape `(r, c)`, return the largest col
CANDIDATES[1370] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """The heaviest column\'s total."""\n    return float(einops.reduce(x, "r c -> c", "sum").max())\n'

# Q1371: Given a 2-D float tensor `counts` of shape `(r, c)` — r rows of non-ne
CANDIDATES[1371] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(counts):\n    """Average share of each column, after every row is made to sum to 1."""\n    shares = counts / einops.reduce(counts, "r c -> r 1", "sum")\n    return einops.reduce(shares, "r c -> 1 c", "mean")\n'

# Q1372: Given a 3-D float tensor `imgs` of shape `(b, h, w)` — a batch of b no
CANDIDATES[1372] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(imgs):\n    """Turn each image into a distribution over its own pixels."""\n    return imgs / einops.reduce(imgs, "b h w -> b 1 1", "sum")\n'

# Q1373: Given a 3-D float tensor `imgs` of shape `(b, h, w)`, return a tuple `
CANDIDATES[1373] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(imgs):\n    """Three summaries of one batch, each over a different set of axes."""\n    per_image = einops.reduce(imgs, "b h w -> b", "sum")\n    per_pixel = einops.reduce(imgs, "b h w -> h w", "mean")\n    overall = float(einops.reduce(imgs, "b h w -> ", "mean"))\n    return per_image, per_pixel, overall\n'

# Q1471: For a 2-D float tensor x, return one sum per row as shape (r,).
CANDIDATES[1471] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.reduce(x, "r c -> r", "sum")\n'

# Q1473: For a 2-D float tensor x, return one mean per row as shape (r,).
CANDIDATES[1473] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.reduce(x, "r c -> r", "mean")\n'

# Q1476: For x shaped (b,h,w), return one mean per image by reducing both spati
CANDIDATES[1476] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.reduce(x, "b h w -> b", "mean")\n'

# Q1477: For x shaped (r,c), return row sums with shape (r,1), keeping reduced 
CANDIDATES[1477] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.reduce(x, "r c -> r 1", "sum")\n'

# Q1481: For any float tensor x, return its overall mean as a Python float.
CANDIDATES[1481] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return float(einops.reduce(x, "... -> 1", "mean").item())\n'

# Q1482: For x shaped (b,c,h,w), return one sum per image while retaining shape
CANDIDATES[1482] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    return einops.reduce(x, "b c h w -> b 1 1 1", "sum")\n'

# =====================================================================
# Subtopic: ar-00 (38 solutions)
# =====================================================================
# Q993: Return row lengths, shape (m,). x: float (m,n), every row nonzero.
CANDIDATES[993] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Return row lengths via einops reduction."""\n    return t.sqrt(einops.reduce(x * x, "m n -> m", "sum"))\n'

# Q994: Return squared row lengths, shape (m, 1). x: float (m,n), every row no
CANDIDATES[994] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Return squared row lengths, shape (m, 1)."""\n    return einops.reduce(x * x, "m n -> m 1", "sum")\n'

# Q995: Return unit rows, shape (m, n). x: float (m,n), every row nonzero.
CANDIDATES[995] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Return unit rows, shape (m, n)."""\n    row_norms = t.sqrt(einops.reduce(x * x, "m n -> m 1", "sum"))\n    return x / row_norms\n'

# Q996: Return each row rescaled to length three, shape (m, n). x: float (m,n)
CANDIDATES[996] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Return each row rescaled to length three."""\n    row_norms = t.sqrt(einops.reduce(x * x, "m n -> m 1", "sum"))\n    return 3 * x / row_norms\n'

# Q997: Return the distance of each row from the origin, shape (m,). x: float 
CANDIDATES[997] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Return distance of each row from origin."""\n    return t.sqrt(einops.reduce(x * x, "m n -> m", "sum"))\n'

# Q998: Return one unit direction per row, shape (m, n). x: float (m,n), every
CANDIDATES[998] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Return one unit direction per row, shape (m, n)."""\n    n = t.sqrt(einops.reduce(x * x, "m n -> m 1", "sum"))\n    return x / t.clamp(n, min=1e-6)\n'

# Q1000: Return the unit direction of the sum of all rows, shape (n,); the sum 
CANDIDATES[1000] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    v = einops.reduce(x, "m n -> n", "sum")\n    return v / v.norm()\n'

# Q1001: Return the unit direction of each row projected onto coordinates 1 thr
CANDIDATES[1001] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    v = x[:, 1:]\n    v_norm = t.sqrt(einops.reduce(v * v, "m n -> m 1", "sum"))\n    return v / v_norm\n'

# Q1003: Return distances between every pair of rows, shape (m,m). x: float (m,
CANDIDATES[1003] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    d = einops.rearrange(x, "m d -> m 1 d") - einops.rearrange(x, "m d -> 1 m d")\n    return d.norm(dim=-1)\n'

# Q1005: Return the length of the average of unit row directions, as a scalar t
CANDIDATES[1005] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    row_norm = t.sqrt(einops.reduce(x * x, "m n -> m 1", "sum"))\n    u = x / row_norm\n    u_mean = einops.reduce(u, "m n -> n", "mean")\n    return t.sqrt(einops.reduce(u_mean * u_mean, "n -> ", "sum"))\n'

# Q1006: Return all row dot products, shape (m,n). x: queries (m,d); y: candida
CANDIDATES[1006] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    return einops.einsum(x, y, "m d, n d -> m n")\n'

# Q1007: Return lengths of every candidate row, shape (n,1). y: candidates (n,d
CANDIDATES[1007] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(y):\n    return einops.rearrange(y.norm(dim=1), "n -> n 1")\n'

# Q1008: Return cosine similarities, shape (m,n). x: queries (m,d); y: candidat
CANDIDATES[1008] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / einops.rearrange(x.norm(dim=1), "m -> m 1")\n    b = y / einops.rearrange(y.norm(dim=1), "n -> n 1")\n    return einops.einsum(a, b, "m d, n d -> m n")\n'

# Q1009: Return the largest cosine similarity per query, shape (m,). x: queries
CANDIDATES[1009] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / t.sqrt(einops.reduce(x * x, "m d -> m 1", "sum"))\n    b = y / t.sqrt(einops.reduce(y * y, "n d -> n 1", "sum"))\n    s = einops.einsum(a, b, "m d, n d -> m n")\n    return s.max(dim=1)[0]\n'

# Q1010: Return cosine similarity of each query to the first candidate, shape (
CANDIDATES[1010] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / t.sqrt(einops.reduce(x * x, "m d -> m 1", "sum"))\n    b = y[0] / t.sqrt(einops.reduce(y[0] * y[0], "d -> ", "sum"))\n    return einops.einsum(a, b, "m d, d -> m")\n'

# Q1011: Return similarities for every candidate against every query, shape (n,
CANDIDATES[1011] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    """Similarities for every candidate against every query, shape (n, m)."""\n    a = x / t.sqrt(einops.reduce(x * x, "m d -> m 1", "sum"))\n    b = y / t.sqrt(einops.reduce(y * y, "n d -> n 1", "sum"))\n    return einops.einsum(b, a, "n d, m d -> n m")\n'

# Q1012: Return nearest candidate indices by cosine similarity, shape (m,); tie
CANDIDATES[1012] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / einops.rearrange(x.norm(dim=1), "m -> m 1")\n    b = y / einops.rearrange(y.norm(dim=1), "n -> n 1")\n    sim = einops.einsum(a, b, "m d, n d -> m n")\n    return sim.argmax(dim=1)\n'

# Q1013: Return each query’s mean similarity to the candidates, shape (m,). x: 
CANDIDATES[1013] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    """Each query\'s mean similarity to candidates."""\n    a = x / t.sqrt(einops.reduce(x * x, "m d -> m 1", "sum"))\n    b = y / t.sqrt(einops.reduce(y * y, "n d -> n 1", "sum"))\n    s = einops.einsum(a, b, "m d, n d -> m n")\n    return einops.reduce(s, "m n -> m", "mean")\n'

# Q1014: Return a Boolean matrix showing strictly opposing directions, shape (m
CANDIDATES[1014] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / einops.rearrange(x.norm(dim=1), "m -> m 1")\n    b = y / einops.rearrange(y.norm(dim=1), "n -> n 1")\n    sim = einops.einsum(a, b, "m d, n d -> m n")\n    return sim < 0\n'

# Q1015: Return one minus cosine similarity for each pair, shape (m,n). x: quer
CANDIDATES[1015] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / einops.rearrange(x.norm(dim=1), "m -> m 1")\n    b = y / einops.rearrange(y.norm(dim=1), "n -> n 1")\n    sim = einops.einsum(a, b, "m d, n d -> m n")\n    return 1 - sim\n'

# Q1016: Return the original candidate vector closest in direction to each quer
CANDIDATES[1016] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    a = x / einops.rearrange(x.norm(dim=1), "m -> m 1")\n    b = y / einops.rearrange(y.norm(dim=1), "n -> n 1")\n    sim = einops.einsum(a, b, "m d, n d -> m n")\n    return y[sim.argmax(dim=1)]\n'

# Q1017: Return Euclidean distances between normalized query and candidate dire
CANDIDATES[1017] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    """Euclidean distances between normalized query and candidate directions."""\n    a = x / t.sqrt(einops.reduce(x * x, "m d -> m 1", "sum"))\n    b = y / t.sqrt(einops.reduce(y * y, "n d -> n 1", "sum"))\n    diff = einops.rearrange(a, "m d -> m 1 d") - einops.rearrange(b, "n d -> 1 n d")\n    return t.sqrt(einops.reduce(diff * diff, "m n d -> m n", "sum"))\n'

# Q1018: Return each query’s similarity to the unit direction of the candidate 
CANDIDATES[1018] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    """Similarity of queries to unit centroid direction of candidates."""\n    a = x / t.sqrt(einops.reduce(x * x, "m d -> m 1", "sum"))\n    c = einops.reduce(y, "n d -> d", "mean")\n    c_norm = c / t.sqrt(einops.reduce(c * c, "d -> ", "sum"))\n    return einops.einsum(a, c_norm, "m d, d -> m")\n'

# Q1032: Return row scores relative to each row’s maximum, shape (b,c). x: logi
CANDIDATES[1032] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    max_val = einops.reduce(x, "b c -> b 1", "max")\n    return x - max_val\n'

# Q1033: Return positive unnormalized weights after removing the largest row sc
CANDIDATES[1033] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = x - einops.reduce(x, "b c -> b 1", "max")\n    return z.exp()\n'

# Q1034: Return normalized class probabilities, shape (b,c). x: logits (b,c), f
CANDIDATES[1034] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Normalized class probabilities (Softmax)."""\n    z = x - x.max(dim=1, keepdim=True)[0]\n    e = z.exp()\n    return e / einops.reduce(e, "b c -> b 1", "sum")\n'

# Q1035: Return row log-normalizers, shape (b,). x: logits (b,c), finite float.
CANDIDATES[1035] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    m = einops.reduce(x, "b c -> b", "max")\n    m_col = einops.rearrange(m, "b -> b 1")\n    s = einops.reduce((x - m_col).exp(), "b c -> b", "sum")\n    return m + s.log()\n'

# Q1036: Return the ratio of each row's largest class probability to its smalle
CANDIDATES[1036] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = x - einops.reduce(x, "b c -> b 1", "max")\n    e = z.exp()\n    p = e / einops.reduce(e, "b c -> b 1", "sum")\n    return einops.reduce(p, "b c -> b", "max") / einops.reduce(p, "b c -> b", "min")\n'

# Q1037: Return log probabilities without built-in log-normalization functions,
CANDIDATES[1037] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Log probabilities without built-in functions."""\n    z = x - x.max(dim=1, keepdim=True)[0]\n    return z - einops.reduce(z.exp(), "b c -> b 1", "sum").log()\n'

# Q1038: Return each row’s largest class probability, shape (b,). x: logits (b,
CANDIDATES[1038] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = x - einops.reduce(x, "b c -> b 1", "max")\n    e = z.exp()\n    p = e / einops.reduce(e, "b c -> b 1", "sum")\n    return einops.reduce(p, "b c -> b", "max")\n'

# Q1039: Return the probability assigned to the first class, shape (b,). x: log
CANDIDATES[1039] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = x - einops.reduce(x, "b c -> b 1", "max")\n    e = z.exp()\n    p = e / einops.reduce(e, "b c -> b 1", "sum")\n    return p[:, 0]\n'

# Q1040: Return the entropy of each row’s distribution, shape (b,); entropy is 
CANDIDATES[1040] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Entropy of each row\'s distribution."""\n    z = x - x.max(dim=1, keepdim=True)[0]\n    l = z - einops.reduce(z.exp(), "b c -> b 1", "sum").log()\n    return -einops.reduce(l.exp() * l, "b c -> b", "sum")\n'

# Q1041: Return the log of the mean exponential score per row, shape (b,). x: l
CANDIDATES[1041] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    m = einops.reduce(x, "b c -> b", "max")\n    m_col = einops.rearrange(m, "b -> b 1")\n    mean_exp = einops.reduce((x - m_col).exp(), "b c -> b", "mean")\n    return m + mean_exp.log()\n'

# Q1042: Return the class probabilities after halving every score difference, s
CANDIDATES[1042] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = (x - einops.reduce(x, "b c -> b 1", "max")) / 2\n    e = z.exp()\n    return e / einops.reduce(e, "b c -> b 1", "sum")\n'

# Q1043: Return the mean class distribution across the batch, shape (c,). x: lo
CANDIDATES[1043] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = x - einops.reduce(x, "b c -> b 1", "max")\n    e = z.exp()\n    p = e / einops.reduce(e, "b c -> b 1", "sum")\n    return einops.reduce(p, "b c -> c", "mean")\n'

# Q1044: Return how much probability each row assigns to scores strictly below 
CANDIDATES[1044] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    z = x - x.max(dim=1, keepdim=True)[0]\n    e = z.exp()\n    p = e / einops.reduce(e, "b c -> b 1", "sum")\n    mean_row = einops.reduce(x, "b c -> b 1", "mean")\n    return einops.reduce(p * (x < mean_row), "b c -> b", "sum")\n'

# Q1049: Return the number of correct predictions, as a scalar tensor. x: logit
CANDIDATES[1049] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    return einops.reduce((x.argmax(dim=1) == y).long(), "b -> ", "sum")\n'

# Q1050: Return the mean negative log probability of the true class, as a scala
CANDIDATES[1050] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, y):\n    z = x - x.max(dim=1, keepdim=True)[0]\n    l = z - einops.reduce(z.exp(), "b c -> b 1", "sum").log()\n    loss = -l[t.arange(len(y)), y]\n    return einops.reduce(loss, "b -> ", "mean")\n'

# =====================================================================
# Subtopic: ar-01 (24 solutions)
# =====================================================================
# Q1101: Return the segment midpoint relative to the ray origin, shape (2,). r:
CANDIDATES[1101] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, s):\n    mid = einops.reduce(s[:, :2], "pts d -> d", "mean")\n    return mid - r[0, :2]\n'

# Q1122: Return all pixel yz coordinates, shape (ny*nz,2). ny: pixels along y; 
CANDIDATES[1122] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    return einops.rearrange([yy, zz], 'two n -> n two')\n"

# Q1123: Return camera rays from the origin through plane x=1, shape (ny*nz,2,3
CANDIDATES[1123] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = yy\n    r[:, 1, 2] = zz\n    return r\n"

# Q1124: Return each camera ray’s direction, shape (ny*nz,3). ny: pixels along 
CANDIDATES[1124] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = yy\n    r[:, 1, 2] = zz\n    return r[:, 1]\n"

# Q1127: Return points reached by these rays at parameter u=2, shape (ny*nz,3).
CANDIDATES[1127] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = yy\n    r[:, 1, 2] = zz\n    return r[:, 0] + 2 * r[:, 1]\n"

# Q1128: Return only rays through the first z-column of the image, shape (ny,2,
CANDIDATES[1128] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = yy\n    r[:, 1, 2] = zz\n    return einops.rearrange(r, '(ny nz) pts d -> ny nz pts d', ny=ny, nz=nz)[:, 0]\n"

# Q1129: Return rays from camera (0,1,0) through the fixed plane x=1, shape (ny
CANDIDATES[1129] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = yy\n    r[:, 1, 2] = zz\n    r[:, 0, 1] = 1\n    r[:, 1, 1] = r[:, 1, 1] - 1\n    return r\n"

# Q1130: Return directions through plane x=2 using the same yz pixel coordinate
CANDIDATES[1130] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, 'y -> (y z)', z=nz)\n    zz = einops.repeat(z, 'z -> (y z)', y=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = yy\n    r[:, 1, 2] = zz\n    r[:, 1, 0] = 2\n    return r[:, 1]\n"

# Q1133: Return directions from camera (0,0,1) to pixels on fixed plane x=1, sh
CANDIDATES[1133] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, "ny -> ny nz", nz=nz)\n    zz = einops.repeat(z, "nz -> ny nz", ny=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = einops.rearrange(yy, "ny nz -> (ny nz)")\n    r[:, 1, 2] = einops.rearrange(zz, "ny nz -> (ny nz)")\n    return r[:, 1] - t.tensor([0.0, 0.0, 1.0])\n'

# Q1134: Return rays from the origin with unit-length directions through plane 
CANDIDATES[1134] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, "ny -> ny nz", nz=nz)\n    zz = einops.repeat(z, "nz -> ny nz", ny=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = einops.rearrange(yy, "ny nz -> (ny nz)")\n    r[:, 1, 2] = einops.rearrange(zz, "ny nz -> (ny nz)")\n    norm = einops.rearrange(r[:, 1].norm(dim=1), "b -> b 1")\n    r[:, 1] = r[:, 1] / norm\n    return r\n'

# Q1135: Return rays through plane x=1 with the image reversed along z within e
CANDIDATES[1135] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, "ny -> ny nz", nz=nz)\n    zz = einops.repeat(z, "nz -> ny nz", ny=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = einops.rearrange(yy, "ny nz -> (ny nz)")\n    r[:, 1, 2] = einops.rearrange(zz, "ny nz -> (ny nz)")\n    image = einops.rearrange(r, "(ny nz) p d -> ny nz p d", ny=ny, nz=nz)\n    rev = image[:, t.arange(nz - 1, -1, -1)]\n    return einops.rearrange(rev, "ny nz p d -> (ny nz) p d")\n'

# Q1136: Return rays from camera (-1,1,0) through fixed plane x=1 with unit-len
CANDIDATES[1136] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(ny, nz, yl, zl):\n    y = t.linspace(-yl, yl, ny)\n    z = t.linspace(-zl, zl, nz)\n    yy = einops.repeat(y, "ny -> ny nz", nz=nz)\n    zz = einops.repeat(z, "nz -> ny nz", ny=ny)\n    r = t.zeros(ny * nz, 2, 3)\n    r[:, 1, 0] = 1\n    r[:, 1, 1] = einops.rearrange(yy, "ny nz -> (ny nz)")\n    r[:, 1, 2] = einops.rearrange(zz, "ny nz -> (ny nz)")\n    o = t.tensor([-1.0, 1.0, 0.0])\n    r[:, 0] = o\n    d = r[:, 1] - o\n    norm = einops.rearrange(d.norm(dim=1), "b -> b 1")\n    r[:, 1] = d / norm\n    return r\n'

# Q1137: Return edge vectors B−A and C−A as columns, shape (3,2). tr: triangle 
CANDIDATES[1137] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(tr):\n    diffs = t.stack((tr[1] - tr[0], tr[2] - tr[0]))\n    return einops.rearrange(diffs, "two d -> d two")\n'

# Q1138: Return the ray/triangle coefficient matrix, shape (3,3). r: ray (2,3) 
CANDIDATES[1138] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    o, d = r\n    a, b, c = tr\n    cols = t.stack((-d, b - a, c - a))\n    return einops.rearrange(cols, "three d -> d three")\n'

# Q1149: Return the centroid of the triangle relative to the ray origin, shape 
CANDIDATES[1149] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    centroid = einops.reduce(tr, "pts d -> d", "mean")\n    return centroid - r[0]\n'

# Q1153: Return hit verdicts for all pairs, shape (nr,nt). r: rays (nr,2,3) as 
CANDIDATES[1153] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    return valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n'

# Q1154: Return travel parameters for valid hits and infinity for misses, shape
CANDIDATES[1154] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    return t.where(hit, s, t.tensor(float("inf")))\n'

# Q1155: Return nearest valid travel parameter per ray, or infinity, shape (nr,
CANDIDATES[1155] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    depth = t.where(hit, s, t.tensor(float("inf")))\n    return einops.reduce(depth, "nr nt -> nr", "min")\n'

# Q1156: Return whether each ray has any visible surface, shape (nr,). r: rays 
CANDIDATES[1156] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    return hit.any(dim=1)\n'

# Q1157: Return number of triangles hit by each ray, shape (nr,). r: rays (nr,2
CANDIDATES[1157] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    return einops.reduce(hit.int(), "nr nt -> nr", "sum")\n'

# Q1158: Return the mean of the nearest travel parameters over the rays that se
CANDIDATES[1158] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    depth = t.where(hit, s, t.tensor(float("inf")))\n    nearest = einops.reduce(depth, "nr nt -> nr", "min")\n    seen = nearest < float("inf")\n    return t.where(seen.any(), nearest[seen].mean(), t.tensor(0.0))\n'

# Q1159: Return index of nearest triangle, or -1 on background, shape (nr,); ti
CANDIDATES[1159] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    depth = t.where(hit, s, t.tensor(float("inf")))\n    value, index = depth.min(dim=1)\n    return t.where(value < float("inf"), index, -1)\n'

# Q1160: Return the index of the triangle crossed by the most rays, counting ev
CANDIDATES[1160] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    counts = einops.reduce(hit.int(), "nr nt -> nt", "sum")\n    return counts.argmax()\n'

# Q1164: Return whether each ray crosses more than one triangle, shape (nr,). r
CANDIDATES[1164] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(r, tr):\n    nr, nt = r.shape[0], tr.shape[0]\n    o = einops.repeat(r[:, 0, :], "nr d -> nr nt d", nt=nt)\n    d = einops.repeat(r[:, 1, :], "nr d -> nr nt d", nt=nt)\n    a = einops.repeat(tr[:, 0, :], "nt d -> nr nt d", nr=nr)\n    b = einops.repeat(tr[:, 1, :], "nt d -> nr nt d", nr=nr)\n    c = einops.repeat(tr[:, 2, :], "nt d -> nr nt d", nr=nr)\n    cols = t.stack((-d, b - a, c - a))\n    m = einops.rearrange(cols, "three nr nt d -> nr nt d three")\n    valid = t.linalg.det(m).abs() >= 1e-8\n    safe = t.where(valid[..., None, None], m, t.eye(3))\n    diff = einops.rearrange(o - a, "nr nt d -> nr nt d 1")\n    suv = t.linalg.solve(safe, diff)[..., 0]\n    s, u, v = suv[..., 0], suv[..., 1], suv[..., 2]\n    hit = valid & (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)\n    counts = einops.reduce(hit.int(), "nr nt -> nr", "sum")\n    return counts > 1\n'

# =====================================================================
# Subtopic: ar-02 (20 solutions)
# =====================================================================
# Q1192: Return the affine output with all leading example axes combined, shape
CANDIDATES[1192] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w, b):\n    y = x @ w.T + b\n    return einops.rearrange(y, '... out_f -> (...) out_f')\n"

# Q1193: Return the mean affine output across all examples and positions, shape
CANDIDATES[1193] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w, b):\n    y = x @ w.T + b\n    return einops.reduce(y, '... out_f -> out_f', 'mean')\n"

# Q1201: Return image features, shape (b,pixels). x: images (b,...), float.
CANDIDATES[1201] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    return einops.rearrange(x, 'b ... -> b (...)')\n"

# Q1202: Return hidden pre-activation values, shape (b,hidden). x: images (b,..
CANDIDATES[1202] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    return flat @ w1.T + b1\n"

# Q1203: Return hidden activations after the positive-part nonlinearity, shape 
CANDIDATES[1203] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    return h\n"

# Q1204: Return classifier logits, shape (b,classes). x: images (b,...), float;
CANDIDATES[1204] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1, w2, b2):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    y = h @ w2.T + b2\n    return y\n"

# Q1205: Return the logits of the two-layer image classifier averaged over the 
CANDIDATES[1205] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1, w2, b2):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    return einops.reduce(h @ w2.T + b2, 'b c -> c', 'mean')\n"

# Q1206: Return predicted class indices, shape (b,); ties choose first. x: imag
CANDIDATES[1206] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1, w2, b2):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    y = h @ w2.T + b2\n    return y.argmax(dim=1)\n"

# Q1207: Return mean hidden activation over the batch, shape (hidden,). x: imag
CANDIDATES[1207] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    return einops.reduce(h, 'b hidden -> hidden', 'mean')\n"

# Q1208: Return number of active hidden features per example, shape (b,); posit
CANDIDATES[1208] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    return (h > 0).sum(dim=1)\n"

# Q1210: Return class probabilities, shape (b,classes). x: images (b,...), floa
CANDIDATES[1210] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1, w2, b2):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h = (flat @ w1.T + b1).clamp(min=0)\n    y = h @ w2.T + b2\n    return y.softmax(dim=-1)\n"

# Q1212: Return the first hidden feature’s contribution to every logit, shape (
CANDIDATES[1212] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, w1, b1, w2):\n    flat = einops.rearrange(x, 'b ... -> b (...)')\n    h0 = (flat @ w1.T + b1)[:, :1].clamp(min=0)\n    return einops.einsum(h0, w2[:, :1], 'b one, classes one -> b classes')\n"

# Q1219: Return m copies of v as a view, shape (m,n), using as_strided. x: floa
CANDIDATES[1219] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x, v):\n    return einops.repeat(v, 'n -> m n', m=x.shape[0])\n"

# Q1222: Return the outer product of v with itself, shape (n,n), using only as_
CANDIDATES[1222] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(v):\n    return einops.einsum(v, v, 'n, m -> n m')\n"

# Q1224: Return a view of shape (m,n,2) whose last axis repeats each source val
CANDIDATES[1224] = "# Solution created by Antigravity\nimport torch as t\nimport einops\n\ndef solve(x):\n    return einops.repeat(x, 'm n -> m n 2')\n"

# Q1303: Return population variance of each input channel, shape (channels,). x
CANDIDATES[1303] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x):\n    """Population variance of each input channel."""\n    mean = einops.reduce(x, "b c h w -> c", "mean")\n    diff = x - einops.rearrange(mean, "c -> 1 c 1 1")\n    return einops.reduce(diff ** 2, "b c h w -> c", "mean")\n'

# Q1304: Return the updated running variance after this batch. x: float (batch,
CANDIDATES[1304] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, rv, momentum):\n    """Updated running variance after this batch."""\n    mean = einops.reduce(x, "b c h w -> c", "mean")\n    diff = x - einops.rearrange(mean, "c -> 1 c 1 1")\n    var = einops.reduce(diff ** 2, "b c h w -> c", "mean")\n    return (1 - momentum) * rv + momentum * var\n'

# Q1309: Return training-mode channel means and population variances after the 
CANDIDATES[1309] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(x, g, b, eps):\n    """Training-mode channel means and population variances after affine transform."""\n    mean = einops.reduce(x, "b c h w -> c", "mean")\n    diff = x - einops.rearrange(mean, "c -> 1 c 1 1")\n    var = einops.reduce(diff ** 2, "b c h w -> c", "mean")\n    z = diff / t.sqrt(einops.rearrange(var, "c -> 1 c 1 1") + eps)\n    y = z * einops.rearrange(g, "c -> 1 c 1 1") + einops.rearrange(b, "c -> 1 c 1 1")\n    ym = einops.reduce(y, "b c h w -> c", "mean")\n    y_diff = y - einops.rearrange(ym, "c -> 1 c 1 1")\n    yv = einops.reduce(y_diff ** 2, "b c h w -> c", "mean")\n    return t.stack((ym, yv))\n'

# Q1312: Implement a stateful BatchNorm module whose forward ends with global a
CANDIDATES[1312] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(g, b, rm, rv, eps, momentum):\n    class BatchNorm(t.nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.weight = t.nn.Parameter(g.clone())\n            self.bias = t.nn.Parameter(b.clone())\n            self.register_buffer("running_mean", rm.clone())\n            self.register_buffer("running_var", rv.clone())\n            self.register_buffer("num_batches_tracked", t.tensor(0))\n        def forward(self, x):\n            if self.training:\n                mean = einops.reduce(x, "b c h w -> c", "mean")\n                diff = x - einops.rearrange(mean, "c -> 1 c 1 1")\n                var = einops.reduce(diff ** 2, "b c h w -> c", "mean")\n                with t.no_grad():\n                    self.running_mean.copy_((1 - momentum) * self.running_mean + momentum * mean)\n                    self.running_var.copy_((1 - momentum) * self.running_var + momentum * var)\n                    self.num_batches_tracked += 1\n            else:\n                mean = self.running_mean\n                var = self.running_var\n                diff = x - einops.rearrange(mean, "c -> 1 c 1 1")\n            z = diff / t.sqrt(einops.rearrange(var, "c -> 1 c 1 1") + eps)\n            out = z * einops.rearrange(self.weight, "c -> 1 c 1 1") + einops.rearrange(self.bias, "c -> 1 c 1 1")\n            return einops.reduce(out, "b c h w -> b c", "mean")\n    return BatchNorm()\n'

# Q1416: Return the classification head as a module: given a feature map `(batc
CANDIDATES[1416] = '# Solution created by Antigravity\nimport einops\nimport torch as t\n\ndef solve(c, n_classes):\n    class Head(t.nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.lin = t.nn.Linear(c, n_classes)\n        def forward(self, x):\n            return self.lin(einops.reduce(x, "b c h w -> b c", "mean"))\n    return Head()\n'

# Now test every candidate
verified = {}
failed = []

print(f"Testing {len(CANDIDATES)} einops candidate solutions...")
for qid, code in sorted(CANDIDATES.items()):
    q = q_by_id.get(qid)
    if not q:
        print(f"Warning: Question {qid} not found in bank!")
        continue
    test_cases = q.get("test_cases", [])
    if not test_cases:
        print(f"Warning: Question {qid} has no test cases!")
        continue
    
    results, exec_res = code_runner.run_function_tests(code, test_cases)
    passed = all(r.passed for r in results)
    if passed:
        verified[str(qid)] = code
        print(f"  ✓ Q{qid} ({q.get('topic')}: {q.get('subtopic')}): PASSED ({len(results)} tests)")
    else:
        failed.append((qid, exec_res.stderr))
        print(f"  ✗ Q{qid}: FAILED - {exec_res.stderr}")

print(f"\nResults: {len(verified)} verified, {len(failed)} failed")

# Save verified solutions to einops_solutions.json
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(verified, f, indent=2)

provenance = {
    "metadata": {
        "generator": "Antigravity (Google DeepMind)",
        "description": "Alternative einops solutions authored and verified by Antigravity",
        "total_solutions": len(verified)
    },
    "solutions": {
        qid: {
            "qid": int(qid),
            "created_by": "Antigravity",
            "model": "Antigravity (Google DeepMind)",
            "topic": q_by_id.get(int(qid), {}).get("topic", ""),
            "subtopic": q_by_id.get(int(qid), {}).get("subtopic", ""),
            "original_answer_source": q_by_id.get(int(qid), {}).get("source_path", "questions.json (Original Curriculum / Previous Model)")
        }
        for qid in verified
    }
}
with open(PROV_PATH, "w", encoding="utf-8") as f:
    json.dump(provenance, f, indent=2)

print(f"Saved {len(verified)} verified solutions to {OUT_PATH} and provenance to {PROV_PATH}")
