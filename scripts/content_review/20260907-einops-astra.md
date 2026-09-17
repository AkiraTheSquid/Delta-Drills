# Delta Drills — einops content review

Standard: [CONTENT_RUBRIC.md](/home/stellar-thread/Applications/Delta-Drills-Local/scripts/CONTENT_RUBRIC.md). Scope: all 12 einops KP pages; only drill rows **847–940**.

**Verdicts: 93 drills major, 1 pass; all 12 lesson pages major.** Major = deploy-blocking under rubric. Verdicts concern supplied content, including bank starters; actual UI selection of lesson versus bank starters remains unverified.

Read-only throughout. NumPy mathematical surrogate matched **211/211 graded outputs** and assertions in all **15 worked examples**. PyTorch unavailable → `CORR_RUNS` remains unverified; surrogate results do not establish backend execution or grader tolerance behavior.

## 1. Drills — per-ID findings

Repeated defects grouped below. Per-ID table maps every failing drill to those issues and supplies concrete replacement wording.

**Pass: 862.** No content defect established within reviewed scope.

### D1 — Invisible cross-references

**`STMT_SELF_CONTAINED` — major.** All 72 IDs below invoke another exercise. Some retain enough subsequent detail to solve; reference still violates rubric explicitly.

| IDs | Exact quote |
|---|---|
| 865–868 | “Variant of `einsum_trace`:” |
| 870–873 | “Variant of `einsum_mv`:” |
| 875–878 | “Variant of `einsum_mm`:” |
| 880–883 | “Variant of `einsum_inner`:” |
| 885–888 | “Variant of `einsum_outer`:” |
| 889–892 | “Variant of exercise (1):” |
| 893–896 | “Variant of exercise (2):” |
| 897–900 | “Variant of exercise (3):” |
| 901–904 | “Variant of exercise (4):” |
| 905–908 | “Variant of exercise (5):” |
| 909–912 | “Variant of exercise (6):” |
| 913–916 | “Variant of exercise (7):” |
| 917–920 | “Variant of exercise (8):” |
| 921–924 | “Variant of (A1):” |
| 925–928 | “Variant of (A2):” |
| 929–932 | “Variant of (B1):” |
| 933–936 | “Variant of (B2):” |
| 937–940 | “Variant of (B3):” |

**Replacement:** use corresponding standalone sentence in per-ID table. Especially 910, 931, 933–935, 937–939 need restored input/grouping contracts; deleting prefix alone leaves omissions.

### D2 — Prompt names implementation being assessed

**`GIVE_PROMPT` — major.**

| IDs | Exact giveaway quote |
|---|---|
| 847, 849, 851, 852, 855–861, 863–865, 868, 869, 874, 879, 884 | “`einops.einsum`” |
| 921, 922 | “build `t.arange(n)`” |
| 923 | “build `t.arange(n)` and split it” |
| 924 | “build `t.arange(start, stop)`” |
| 925 | “with `einops.rearrange` on `t.tensor(vals)`” |
| 926 | “with one `einops.rearrange` call” |
| 928 | “with `einops.rearrange`” |
| 929 | “via `einops.reduce`” |
| 933 | “reduce, repeat back, subtract” |
| 936 | “reduce over days and repeat back along the day axis” |
| 937 | “Use `einops.reduce` for the min and max and `einops.repeat` to lay them back out.” |
| 938 | “pass `t.std` as the reduction to `einops.reduce`” |
| 939 | “using `t.std` as the reduction” |
| 940 | “std via `t.std`” |

Function names matter here: `einops.einsum`, `einops.rearrange`, `einops.reduce`, and `einops.repeat` are declared learning targets. Naming them supplies a choice rubric requires learner to make.

**Replacement:** corresponding goal sentence in per-ID table. For 938–940, specify **sample standard deviation mathematically**, replacing dependence on library defaults.

### D3 — Starter docstrings reveal move

**`GIVE_STARTER` — major.**

| IDs | Exact quote | Replacement docstring sentence |
|---|---|---|
| 847 | “Column sums via einsum.” | “Return one total per matrix column.” |
| 848 | “Sum the last axis of a 3-D tensor via einsum.” | “Return one total for each innermost list.” |
| 849 | “Vector times matrix via einsum.” | “Return the weighted sum of the matrix rows.” |
| 850 | “A times B-transpose via einsum.” | “Return the dot product of every row of the first matrix with every row of the second.” |
| 851 | “The diagonal via a repeated index.” | “Return the main-diagonal entries in order.” |
| 860 | “Row scaling as an einsum.” | “Multiply each row by its corresponding weight.” |
| 864 | “einsum_trace” | “Return the sum of the main-diagonal entries.” |
| 869 | “einsum_mv” | “Return the dot product of each matrix row with the vector.” |
| 874 | “einsum_mm” | “Return the matrix product.” |
| 879 | “einsum_inner” | “Return the sum of corresponding vector-entry products.” |
| 884 | “einsum_outer” | “Return every product between an entry of the first vector and an entry of the second.” |
| 921 | “arange into h rows” | “Arrange the integers from zero through n minus one into h rows.” |
| 924 | “arange into w columns” | “Arrange the integers from start through stop minus one into w columns.” |

### D4 — Faded bank starters lack scaffolds

**`GIVE_RUNG` — major. IDs: 847–854.**

Exact bank-starter quote, all eight:

```python
return None
```

Frontmatter assigns all eight to `faded`. Bank starters contain no blanks or tensor-conversion scaffold; page supplies separate faded versions.

**Replacement:** use each ID’s existing lesson `python starter` fence as bank starter, with D3 docstring corrections. For 847, concrete replacement:

```python
def solve(mat):
    """Return one total per matrix column."""
    a = t.tensor(mat)
    return einops._____(a, "_____").tolist()
```

Apply matching one-/two-operand scaffold to other IDs; preserve `.item()` for scalar-returning 852. This fixes supplied bank content; UI exposure cannot be inferred from these files.

### D5 — No edge case

**`STMT_CASES` — major. IDs: 847, 849, 858, 865–871, 874–876, 879, 880, 882, 883, 888, 905, 934, 938.**

These 21 case sets have no size-one dimension, single block, or other demonstrated boundary. Existing outputs differ; defect concerns required edge coverage.

Each row quotes an existing case and gives exact replacement/addition wording.

| ID | Exact existing `call` quote | Concrete case addition |
|---|---|---|
| 847 | `solve([[0, 1, 2], [3, 4, 5]])` | “For `mat=[[2,-1,4]]`, return `[2,-1,4]`.” |
| 849 | `solve([1, 0], [[0, 1, 2], [3, 4, 5]])` | “For `vec=[3]` and `mat=[[2,-1]]`, return `[6,-3]`.” |
| 858 | `solve([[1, 2], [3, 4]], [[1, 2], [3, 4]])` | “For `mat1=[[2,-1]]` and `mat2=[[3,4]]`, return `[2]`.” |
| 865 | `solve([[1, 2], [3, 4]])` | “For `mat=[[3]]`, return `9`.” |
| 866 | `solve([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])` | “For `mats=[[[3]]]`, return `[3]`.” |
| 867 | `solve([[1, 2], [3, 4]])` | “For `mat=[[-3]]`, return `9`.” |
| 868 | `solve([[1, 2], [3, 4]], -1)` | “For `mat=[[3]]` and `s=-2`, return `[-6]`.” |
| 869 | `solve([[1, 2], [3, 4]], [1, 0])` | “For `mat=[[2,-1]]` and `vec=[3,4]`, return `[2]`.” |
| 870 | `solve([1, 1], [[1, 2], [3, 4]])` | “For `vec=[3]` and `mat=[[2,-1]]`, return `[6,-3]`.” |
| 871 | `solve([[1, 2], [3, 4]], [1, 0])` | “For `mat=[[2,-1]]` and `vec=[3]`, return `[6,-3]`.” |
| 874 | `solve([[1, 2], [3, 4]], [[1, 2], [3, 4]])` | “For `mat1=[[2,3]]` and `mat2=[[4],[5]]`, return `[[23]]`.” |
| 875 | `solve([[1, 2], [3, 4]], [[1, 2], [3, 4]])` | “For `mat1=[[2,3]]` and `mat2=[[4,5]]`, return `[[23]]`.” |
| 876 | `solve([[1, 2], [3, 4]], [[1, 1], [0, 1]])` | “For `mat1=[[2],[3]]` and `mat2=[[4],[5]]`, return `[[23]]`.” |
| 879 | `solve([1, -1], [1, 1])` | “For `vec1=[3]` and `vec2=[-2]`, return `-6`.” |
| 880 | `solve([3, 4])` | “For `vec=[-3]`, return `9`.” |
| 882 | `solve([2, 2], [1, 1], [1, -1])` | “For `w=[2]`, `a=[3]`, and `b=[-4]`, return `-24`.” |
| 883 | `solve([[1, 2], [3, 4]], [[1, 2], [3, 4]])` | “For `mat1=[[3]]` and `mat2=[[-2]]`, return `-6`.” |
| 888 | `solve([1, 1], [1, 1], [[0, 1], [1, 0]])` | “For `vec1=[2]`, `vec2=[3,4]`, and `mask=[[0,1]]`, return `[[0,8]]`.” |
| 905 | `solve([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])` | “For `img=[[[1,2],[3,4]]]`, return `[[1,2],[3,4]]`.” |
| 934 | `solve([1.0, 3.0, 5.0, 7.0], 2)` | “For `temps=[2.0,4.0,6.0]` and `k=3`, return `[-2.0,0.0,2.0]`.” |
| 938 | `solve([1.0, 3.0, 2.0, 4.0], 2)` | “For `temps=[1.0,3.0]` and `k=2`, return `[-0.7071,0.7071]`.” |

### D6 — Pooling cases fail to distinguish mapping

**847–940 mathematical checks found no incorrect stored expected result. These three case sets nevertheless admit relevant incorrect mappings.**

| IDs | Codes / severity | Exact quote and evidence | Replacement case sentence |
|---|---|---|---|
| 917 | `STMT_CASES`, `CORR_DECISIVE` — **major** | Near-miss explanation names “`(h2 h)`”. Every graded input has height 2; swapping the height factors produces **correct output on both cases**. | “For `img=[[[1,2],[3,4],[9,8],[7,6]]]`, return `[[[4],[9]]]`; pooling separated row groups instead gives `[[[9],[7]]]`.” |
| 918 | `STMT_CASES` — **major** | Calls are `solve([[[1.0, 3.0], [5.0, 7.0]]])` and `solve([[[2.0, 2.0, 4.0, 4.0], [2.0, 2.0, 4.0, 4.0]]])`. Height is always one window, leaving height-group order untested. | “For `img=[[[1.0,2.0],[3.0,4.0],[9.0,8.0],[7.0,6.0]]]`, return `[[[2.5],[7.5]]]`.” |
| 920 | `STMT_CASES` — **major** | Expected outputs are `"[[[3, 7]]]"` and `"[[[4, 8]]]"`. Every image pools to one pixel; concatenation and interleaving coincide. | “For two images whose pooled rows are `[4,8]` and `[40,80]`, return `[[[4,8,40,80]]]`, preserving each image’s complete row.” |

For 918 and 920, supplied max-versus-mean / vertical-versus-horizontal near misses do fail. Their finding is broader mapping coverage under `STMT_CASES`, not a claim that those supplied near misses pass.

### D7 — Near-miss explanations contain false claims

**`STMT_NEAR_MISS` — minor.**

| ID | Exact quote | Rewritten replacement sentence |
|---|---|---|
| 910 | “with 4 images and cols=2 the shapes coincide but the layout is the transpose of what was asked.” | “Passing `cols` as the row count makes the `cols=4` example a vertical stack; when four images use `cols=2`, this particular mistake produces the correct layout and cannot be detected by that case.” |
| 917 | “`(h2 h)` pools rows that are h apart (top half against bottom half), not neighbouring rows.” | “The shown output `[[[7,8]]]` comes from grouping nonadjacent columns; this two-row input cannot expose an incorrect ordering of the height factors.” |
| 929 | “`(7 h)` groups every h-th day together (all Mondays, all Tuesdays…), not consecutive weeks.” | “With two weeks, this grouping combines alternating readings; weekday groups instead contain readings seven positions apart.” |
| 934 | “Without the repeat the shapes do not line up — broadcasting fails (or silently pairs the wrong days when the lengths coincide).” | “Subtracting the two block means directly from this 14-element list fails because lengths 14 and 2 cannot broadcast; each mean must be aligned with the readings in its own block.” |

### D8 — Incomplete or conflicting input/output contracts

| IDs | Codes / severity | Exact quote | Concrete correction |
|---|---|---|---|
| 856 | `STMT_OUTPUT` — **major** | “Return the **Gram matrix** `A @ A.T`” | “Return a nested list of shape `(m,m)`, where entry `[i][j]` is the dot product of rows `i` and `j` of `mat`.” |
| 885 | `STMT_OUTPUT` — **major** | “entry `[j, i]` is `vec1[i] * vec2[j]`, shape (n, m).” | “Given lists `vec1` and `vec2` of lengths `m` and `n`, return an `(n,m)` nested list whose entry `[j][i]` is `vec1[i] * vec2[j]`.” |
| 888 | `STMT_OUTPUT` — **major** | “entry `[i, j]` is `vec1[i] * vec2[j] * mask[i][j]`, one call.” | “Return an `(m,n)` nested list whose entry `[i][j]` is `vec1[i] * vec2[j] * mask[i][j]`.” |
| 910 | `STMT_INPUT` — **major** | “same grid, but the number of COLUMNS is given” | “Given a nested list `imgs` of shape `(b,c,h,w)` and a positive integer `cols` dividing `b`, arrange images left to right across each grid row, then continue on the next row.” |
| 911 | `STMT_OUTPUT` — **major** | “into a grid with `rows` rows, shape `(rows·h, cols·w)`” | “Place images in input order, left to right and then top to bottom, in a grid with `rows` rows and `cols=b/rows` columns.” |
| 918, 920 | `STMT_INPUT` — **major** | “shape `(c, h/2, w/2)`”; “shape `(c, h/2, b·w/2)`” | “Height and width are positive even integers; use disjoint 2×2 windows covering the image.” |
| 919 | `STMT_INPUT` — **major** | “by a factor `k` in both directions” | “The positive integer `k` divides both image dimensions, and windows cover the image without overlap or padding.” |
| 921, 922 | `STMT_INPUT` — **major** | “`h` rows”; “`h` given” | “The positive integers `n` and `h` satisfy `n % h == 0`; the output has `n/h` columns.” |
| 923 | `STMT_INPUT` — **major** | “with `a` and `b` given, row-major” | “The positive integers `a` and `b` satisfy `n % (a*b) == 0`, and the final dimension has length `n/(a*b)`.” |
| 924 | `STMT_INPUT` — **major** | “into `w` COLUMNS” | “The arguments are integers with `start < stop`, `w > 0`, and `(stop-start) % w == 0`.” |
| 925, 926 | `STMT_INPUT` — **major** | “split it into `h` rows”; “split into rows of length `w`” | “The list is nonempty, and the positive requested row count or row length divides its length exactly.” |
| 927 | `STMT_INPUT`, `STMT_OUTPUT` — **major** | “split it into `b` matrices of `h` rows each” | “The positive integers `b` and `h` divide the list length jointly; fill each matrix row by row before starting the next matrix, with `w=len(vals)/(b*h)`.” |
| 930, 934 | `STMT_INPUT` — **major** | “blocks of `k` days”; “blocks are `k` days long” | “The input is a nonempty flat list of floating-point readings, and positive integer `k` divides its length; blocks are consecutive runs starting at index zero.” |
| 931, 933 | `STMT_INPUT` — **major** | “each week's readings”; “for each day” | “The input is a nonempty flat list containing a whole number of consecutive seven-day weeks, beginning at index zero.” |
| 936 | `STMT_INPUT`, `STMT_CASES` — **major** | Prompt: “`temps` is `(weeks, 7)`”; graded call: `solve([[2.0, 4.0]])` | “For the single-week input `[[1.0,2.0,3.0,4.0,5.0,6.0,7.0]]`, return `[[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]]`.” |

936 correction preserves stated seven-day contract. Alternatively generalizing to arbitrary block width would require rewriting task consistently.

### D9 — Division domain missing

**`STMT_INPUT` — major. IDs: 935, 937–940.**

| IDs | Exact quote | Rewritten constraint sentence |
|---|---|---|
| 935 | “day / weekly mean” | “The input contains complete consecutive seven-day weeks, and every week has a nonzero arithmetic mean.” |
| 937 | “`(day - week_min) / (week_max - week_min)`” | “The input contains complete consecutive seven-day weeks, and each week contains at least two distinct readings.” |
| 938 | “`(day - block_mean) / block_std`” | “The flat input length is divisible by integer `k≥2`, and every consecutive block of `k` readings has nonzero sample standard deviation.” |
| 939 | “`day / week_std`” | “The flat input contains complete consecutive seven-day weeks, each with nonzero sample standard deviation.” |
| 940 | “the z-score of each day within its week” | “Every row contains seven readings with nonzero sample standard deviation.” |

No new zero-denominator behavior prescribed here: smallest repair is explicit domain restriction matching current answers. Supporting constant blocks instead would require a separately specified result and corresponding answer/case changes.

**`STMT_TERMS` — minor. IDs: 938–940.**

Exact phrases: “using `t.std` for the std”, “using `t.std` as the reduction”, “std via `t.std`”.

Replacement: “Sample standard deviation is the square root of the sum of squared deviations from the block mean divided by one less than the block length.”

### D10 — Basic moves assigned integrated rung

**`GIVE_RUNG` — major; `DIFF_RUNG` — minor.**

IDs: **864, 869, 874, 879, 880, 883, 884, 889–891, 893, 894, 896, 901, 902, 905, 906, 913–916, 921, 924, 925, 928, 932.**

Representative exact quotes:

- 864: “return the trace of a square matrix”.
- 896: “give it 3 identical colour channels”.
- 913: “transpose EVERY image of a batch”.
- 925: “split it into `h` rows”.
- 932: “return the mean of each DAY-OF-WEEK across all weeks”.

These are direct instances of one taught operation. Higher rank, context changes, or converting a list to a tensor do not create an additional KP idea.

**Concrete repair:** assign these existing tasks to Solo; use standalone replacement sentences below. If integrated slots must remain, replace task itself. Example replacement for 913:

> “Transpose each image’s spatial dimensions, then arrange the batch in a row-major grid with a supplied row count, returning one channels-first image.”

**`DIFF_RUNG` — minor. ID 855.**

Quote: “Sum a `(b, i, j)` nested list over its FIRST axis”.

Operation and surviving axes already determined. Replacement faded task sentence:

> “Return the elementwise total of all matrices in `x` as an `(i,j)` nested list.”

Pair with scaffold requiring missing operation and pattern.

### D11 — Duplicate move on same rung

**`DIFF_DUP` — minor. IDs: 863, 875.**

Exact quotes:

- 863: “dot products between every query and every key”.
- 875: “`A @ B.T` — `mat1` is (m, k), `mat2` is (n, k).”

Both integrated items compute identical mapping from two matrices: all row-pair dot products. Attention naming supplies context, not a different operation.

**Replacement for 875:**

> “Given batches `mat1` and `mat2` of shapes `(b,m,d)` and `(b,n,d)`, return a `(b,m,n)` nested list of row-pair dot products, pairing matrices only within the same batch entry.”

### D12 — Custom reduction not taught

**`EXPL_PREREQ` — major. IDs: 938–940.**

Exact answer-code evidence:

```python
einops.reduce(x, '(h k) -> h', t.std, k=k)
```

Owning repeat page and supplied reduce page never teach a callable reduction or its `(tensor, reduced_axes)` interface. Prompt supplies implementation directly instead of lesson establishing prerequisite.

**Replacement lesson sentence:**

> “A custom reduction receives the tensor and a tuple of axes to collapse; for sample standard deviation, aggregate those axes using the block length minus one in the variance denominator.”

Follow with a short worked custom-reduction example on a different axis/input, then retain implementation-free drill wording below.

### D13 — Worked-example reuse

**`GIVE_EXAMPLE` — major. IDs: 847, 903.**

| ID | Exact lesson quote | Collision | Concrete repair sentence |
|---|---|---|---|
| 847 | `a = t.arange(6).reshape(2, 3)` and `assert cols.tolist() == [3, 5, 7]` | Same matrix and expected column sums as first graded case. | “Given a nested list of shape `(batch,rows,cols)`, return one column-total vector per batch entry, preserving the batch axis.” |
| 903 | `img = t.tensor([[1, 2],` followed by `[3, 4]])`; `up = einops.repeat(img, 'h w -> (h a) (w b)', a=2, b=2)` | Worked upscale reproduces first graded input and full output. | “Demonstrate row-only enlargement on a different image, then ask the drill to enlarge both spatial dimensions.” |

For 847, proposed transfer task requires updated scaffold/cases; alternatively remove data collision while separately fixing lesson’s transfer defect. These findings identify source pairing collisions, not inferred runtime example scheduling.

### D14 — Unhelpful or inconsistent wording

| IDs | Code / severity | Exact quote | Rewritten replacement sentence |
|---|---|---|---|
| 897 | `STMT_VOICE` — **note** | “and each stretched? No:” | “Place the images side by side, then place a second complete copy of the resulting strip directly below it.” |
| 906 | `STMT_TERMS` — **minor** | “channels side by side but with the channel index INNERMOST” | “Interleave the channels’ columns: column zero from every channel comes first, followed by column one from every channel.” |
| 908 | `STMT_TERMS` — **minor** | “entry `[x, c·H + y] = img[c][y][x]`” | “For channel index `k`, input row `y`, and input column `x`, output entry `[x][k*h+y]` equals `img[k][y][x]`.” |
| 880 | `STMT_TERMS` — **minor** | “the squared L2 norm of a vector” | “Return the sum of the squares of all vector entries as one number.” |

### Per-ID verdicts and replacement sentences

All rows below: **major**, owing to at least one major issue referenced. D-number references identify grouped findings above; minor/note findings remain separately marked there. Replacement wording fixes statements; case, scaffold, prerequisite, and rung repairs still apply.

#### 847–888

| ID | Findings | Rewritten replacement |
|---|---|---|
| 847 | D2, D3, D4, D5, D13 | “Return one sum per column of the nonempty rectangular nested list `mat`, preserving column order.” |
| 848 | D3, D4 | “Given a nested list `x` of shape `(b,i,j)`, return an `(b,i)` nested list containing the sum of each innermost list.” |
| 849 | D2, D3, D4, D5 | “Given a length-`m` list `vec` and an `(m,n)` nested list `mat`, return the length-`n` weighted sum of the matrix rows, using `vec` as the row weights.” |
| 850 | D3, D4 | “Given nested lists `mat1` and `mat2` of shapes `(m,k)` and `(n,k)`, return an `(m,n)` nested list containing every pairwise dot product between their rows.” |
| 851 | D2, D3, D4 | “Return the main-diagonal entries of square nested list `mat` as a list, ordered from top left to bottom right.” |
| 852 | D2, D4 | “Given two nested lists of the same matrix shape, return the sum of the products of corresponding entries as one number.” |
| 853 | D4 | “Given nested lists `mats` and `vecs` of shapes `(b,i,j)` and `(b,j)`, return an `(b,i)` nested list of matrix–vector products, pairing entries with the same batch index.” |
| 854 | D4 | “Given nested lists `u` and `v` of shapes `(b,n)` and `(b,m)`, return an `(b,n,m)` nested list whose entry `[s][i][j]` is `u[s][i]*v[s][j]`.” |
| 855 | D2, D10 | “Return the elementwise total of the matrices in nested list `x`, converting shape `(b,i,j)` to `(i,j)`.” |
| 856 | D2, D8 | “Given an `(m,n)` nested list `mat`, return an `(m,m)` nested list whose entry `[i][j]` is the dot product of rows `i` and `j`.” |
| 857 | D2 | “Given lists `x` and `y` of lengths `m` and `n`, and an `(m,n)` nested list `mat`, return the scalar bilinear form `x @ mat @ y`.” |
| 858 | D2, D5 | “Given two `(m,n)` nested lists, return a length-`m` list containing the dot product of corresponding rows.” |
| 859 | D2 | “Given a nested list `mats` of shape `(b,n,n)`, return the main diagonal of each matrix as an `(b,n)` nested list.” |
| 860 | D2, D3 | “Given an `(m,n)` nested list `mat` and length-`m` list `w`, multiply every entry in row `i` by `w[i]`, returning an `(m,n)` nested list.” |
| 861 | D2 | “Given nested lists `mat1` and `mat2` of shapes `(m,n)` and `(n,m)`, return the sum of the main-diagonal entries of their matrix product as one number, without constructing the full product.” |
| 863 | D2, D11 | “Given query and key matrices `q` and `k` as nested lists of shapes `(nq,d)` and `(nk,d)`, return an `(nq,nk)` nested list containing every query–key dot product.” |
| 864 | D2, D3, D10 | “Return the sum of the main-diagonal entries of square nested list `mat` as one number.” |
| 865 | D1, D2, D5 | “Given square nested list `mat`, return the sum of the main-diagonal entries of `mat @ mat` as one number, without constructing the full product.” |
| 866 | D1, D5 | “Given a nested list `mats` of shape `(b,n,n)`, return a length-`b` list containing the sum of each matrix’s main-diagonal entries.” |
| 867 | D1, D5 | “Given square nested list `mat`, return the sum of the squares of its main-diagonal entries as one number.” |
| 868 | D1, D2, D5 | “Given square nested list `mat` and scalar `s`, return its main-diagonal entries multiplied by `s` as a list.” |
| 869 | D2, D3, D5, D10 | “Given an `(m,n)` nested list `mat` and length-`n` list `vec`, return a length-`m` list containing the dot product of each matrix row with `vec`.” |
| 870 | D1, D5 | “Given length-`m` list `vec` and `(m,n)` nested list `mat`, return the length-`n` weighted sum of the matrix rows.” |
| 871 | D1, D5 | “Given `(m,n)` nested list `mat` and length-`m` list `vec`, return a length-`n` list containing the dot product of each matrix column with `vec`.” |
| 872 | D1 | “Given `(m,n)` nested list `mat` and `(b,n)` nested list `vecs`, return an `(b,m)` nested list containing the product of `mat` with each vector, preserving vector order.” |
| 873 | D1 | “Given `(b,m,n)` nested list `mats` and length-`n` list `vec`, return an `(b,m)` nested list containing the product of each matrix with `vec`.” |
| 874 | D2, D3, D5, D10 | “Given nested lists `mat1` and `mat2` of shapes `(m,n)` and `(n,p)`, return their matrix product as an `(m,p)` nested list.” |
| 875 | D1, D5, D11 | “Given nested lists `mat1` and `mat2` of shapes `(m,k)` and `(n,k)`, return an `(m,n)` nested list of pairwise row dot products.” |
| 876 | D1, D5 | “Given nested lists `mat1` and `mat2` of shapes `(k,m)` and `(k,n)`, return an `(m,n)` nested list containing every pairwise dot product between their columns.” |
| 877 | D1 | “Given nested lists `mats1` and `mats2` of shapes `(b,m,n)` and `(b,n,p)`, return an `(b,m,p)` nested list of matrix products, pairing matrices with the same batch index.” |
| 878 | D1 | “Given nested-list matrices `a`, `b`, and `c` of shapes `(m,n)`, `(n,p)`, and `(p,q)`, return their ordered matrix product as an `(m,q)` nested list.” |
| 879 | D2, D3, D5, D10 | “Given equal-length lists `vec1` and `vec2`, return the sum of the products of corresponding entries as one number.” |
| 880 | D1, D5, D10, D14 | “Given list `vec`, return the sum of the squares of its entries as one number.” |
| 881 | D1 | “Given nested lists `u` and `v` of shape `(b,n)`, return a length-`b` list containing the dot product of each corresponding pair of rows.” |
| 882 | D1, D5 | “Given equal-length lists `w`, `a`, and `b`, return the weighted dot product—the sum of `w[i]*a[i]*b[i]` over all indices—as one number.” |
| 883 | D1, D5, D10 | “Given same-shape nested-list matrices `mat1` and `mat2`, return the sum of the products of corresponding entries as one number.” |
| 884 | D2, D3, D10 | “Given lists `vec1` and `vec2` of lengths `m` and `n`, return an `(m,n)` nested list whose entry `[i][j]` is `vec1[i]*vec2[j]`.” |
| 885 | D1, D8 | “Given lists `vec1` and `vec2` of lengths `m` and `n`, return an `(n,m)` nested list whose entry `[j][i]` is `vec1[i]*vec2[j]`.” |
| 886 | D1 | “Given nested lists `u` and `v` of shapes `(b,m)` and `(b,n)`, return an `(b,m,n)` nested list whose entry `[s][i][j]` is `u[s][i]*v[s][j]`.” |
| 887 | D1 | “Given lists `a`, `b`, and `c` of lengths `l`, `m`, and `n`, return an `(l,m,n)` nested list whose entry `[i][j][k]` is `a[i]*b[j]*c[k]`.” |
| 888 | D1, D5, D8 | “Given lists `vec1` and `vec2` of lengths `m` and `n` and an `(m,n)` nested list `mask`, return an `(m,n)` nested list whose entry `[i][j]` is `vec1[i]*vec2[j]*mask[i][j]`.” |

#### 889–920

| ID | Findings | Rewritten replacement |
|---|---|---|
| 889 | D1, D10 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, place complete images side by side in batch order, returning a nested list of shape `(c,h,b*w)`.” |
| 890 | D1, D10 | “Given nested-list images `imgs` of shape `(b,h,w,c)`, place complete images vertically in batch order, returning a nested list of shape `(b*h,w,c)`.” |
| 891 | D1, D10 | “Given a nested-list image `img` of shape `(c,h,w)`, place channel zero’s complete image above channel one’s, continuing in channel order, and return a nested list of shape `(c*h,w)`.” |
| 892 | D1 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, return a `(c,b*h*w)` nested list; for each channel, read each image row by row before continuing to the next image.” |
| 893 | D1, D10 | “Given a nested-list image `img` of shape `(c,h,w)`, place a complete second copy to its right, returning shape `(c,h,2*w)`.” |
| 894 | D1, D10 | “Given a nested-list image `img` of shape `(c,h,w)` and positive integer `n`, stack `n` complete copies vertically, returning shape `(c,n*h,w)`.” |
| 895 | D1 | “Given a grayscale nested-list image `img` of shape `(h,w)`, place four complete copies in a two-row, two-column grid, returning shape `(2*h,2*w)`.” |
| 896 | D1, D10 | “Given a grayscale nested-list image `img` of shape `(h,w)`, return three identical channel planes as a nested list of shape `(3,h,w)`.” |
| 897 | D1, D14 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, place them side by side in batch order, then place a complete second copy of that strip below it, returning shape `(c,2*h,b*w)`.” |
| 898 | D1 | “Given nested-list images `imgs` of shape `(b,c,h,w)` and positive integer `n`, stack images vertically in batch order and place `n` complete copies of that strip side by side, returning shape `(c,b*h,n*w)`.” |
| 899 | D1 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, stack complete images vertically in the order `0,0,1,1,…`, returning shape `(c,2*b*h,w)`.” |
| 900 | D1 | “Given grayscale nested-list images `imgs` of shape `(b,h,w)`, stack images vertically in batch order and place a complete copy to the right, returning shape `(b*h,2*w)`.” |
| 901 | D1, D10 | “Given a nested-list image `img` of shape `(c,h,w)`, replace each column with two adjacent identical columns, returning shape `(c,h,2*w)`.” |
| 902 | D1, D10 | “Given a nested-list image `img` of shape `(c,h,w)` and positive integer `k`, replace each row with `k` consecutive identical rows, returning shape `(c,k*h,w)`.” |
| 903 | D1, D13 | “Given a grayscale nested-list image `img` of shape `(h,w)`, replace each pixel with a 2×2 block of the same value, returning shape `(2*h,2*w)`.” |
| 904 | D1 | “Given a nested-list image `img` of shape `(c,h,w)`, replace each pixel in each channel with a block two rows high and three columns wide, returning shape `(c,2*h,3*w)`.” |
| 905 | D1, D5, D10 | “Given a nested-list image `img` of shape `(c,h,w)`, output row zero from every channel, then row one from every channel, continuing in order, as a nested list of shape `(h*c,w)`.” |
| 906 | D1, D10, D14 | “Given a nested-list image `img` of shape `(c,h,w)`, interleave columns in channel order—column zero from every channel, then column one from every channel—returning shape `(h,w*c)`.” |
| 907 | D1 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, place each image’s complete channel planes side by side in channel order, preserving the batch axis and returning shape `(b,h,c*w)`.” |
| 908 | D1, D14 | “Given a nested-list image `img` of shape `(c,h,w)`, transpose each channel plane and place the results side by side in channel order, returning shape `(w,c*h)`.” |
| 909 | D1 | “Given nested-list images `imgs` of shape `(b,c,h,w)` and positive integer `rows` dividing `b`, fill a grid left to right and then top to bottom, returning shape `(c,rows*h,(b/rows)*w)`.” |
| 910 | D1, D7, D8 | “Given nested-list images `imgs` of shape `(b,c,h,w)` and positive integer `cols` dividing `b`, fill a grid left to right and then top to bottom, returning shape `(c,(b/cols)*h,cols*w)`.” |
| 911 | D1, D8 | “Given grayscale nested-list images `imgs` of shape `(b,h,w)` and positive integer `rows` dividing `b`, fill a grid left to right and then top to bottom, returning shape `(rows*h,(b/rows)*w)`.” |
| 912 | D1 | “Given nested-list images `imgs` of shape `(b,c,h,w)` and positive integer `rows` dividing `b`, fill each grid column from top to bottom before moving right, returning shape `(c,rows*h,(b/rows)*w)`.” |
| 913 | D1, D10 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, transpose each channel plane while preserving batch and channel order, returning shape `(b,c,w,h)`.” |
| 914 | D1, D10 | “Given a nested-list image `img` of shape `(c,h,w)`, return its unchanged pixel values in channels-last layout `(h,w,c)`.” |
| 915 | D1, D10 | “Given a nested-list image `img` of shape `(h,w,c)`, return its unchanged pixel values in channels-first layout `(c,h,w)`.” |
| 916 | D1, D10 | “Given nested-list images `imgs` of shape `(b,c,h,w)`, group the whole batch under each channel while preserving spatial coordinates, returning shape `(c,b,h,w)`.” |
| 917 | D1, D6, D7 | “Given a nested-list image `img` of shape `(c,h,w)` with positive even height and width, replace each disjoint 2×2 block in each channel with its maximum, returning shape `(c,h/2,w/2)`.” |
| 918 | D1, D6, D8 | “Given a floating-point nested-list image `img` of shape `(c,h,w)` with positive even height and width, replace each disjoint 2×2 block in each channel with its arithmetic mean, returning shape `(c,h/2,w/2)`.” |
| 919 | D1, D8 | “Given a grayscale nested-list image `img` of shape `(h,w)` and positive integer `k` dividing both dimensions, replace each disjoint `k×k` block with its minimum, returning shape `(h/k,w/k)`.” |
| 920 | D1, D6, D8 | “Given nested-list images `imgs` of shape `(b,c,h,w)` with positive even height and width, replace each disjoint 2×2 block with its maximum and place complete resulting images side by side in batch order, returning shape `(c,h/2,b*(w/2))`.” |

#### 921–940

| ID | Findings | Rewritten replacement |
|---|---|---|
| 921 | D1, D2, D3, D8, D10 | “Given positive integers `n` and `h`, with `h` dividing `n`, return the integers from `0` through `n-1` in an `(h,n/h)` nested list, filling each row from left to right before starting the next.” |
| 922 | D1, D2, D8 | “Given positive integers `n` and `h`, with `h` dividing `n`, return the integers from `0` through `n-1` in an `(h,n/h)` nested list, filling each column from top to bottom before starting the next.” |
| 923 | D1, D2, D8 | “Given positive integers `n`, `a`, and `b`, with `a*b` dividing `n`, return the integers from `0` through `n-1` in shape `(a,b,n/(a*b))`, varying the last index fastest.” |
| 924 | D1, D2, D3, D8, D10 | “Given integers `start < stop` and positive integer `w` dividing `stop-start`, return the integers from `start` through `stop-1` in rows of length `w`, filling each row before starting the next.” |
| 925 | D1, D2, D8, D10 | “Given a nonempty flat list `vals` and positive integer `h` dividing its length, return an `(h,len(vals)/h)` nested list, preserving order and filling each row before starting the next.” |
| 926 | D1, D2, D8 | “Given a nonempty flat list `vals` and positive integer `w` dividing its length, interpret consecutive groups of `w` entries as rows and return the transposed grid as a `(w,len(vals)/w)` nested list.” |
| 927 | D1, D8 | “Given a nonempty flat list `vals` and positive integers `b` and `h`, with `b*h` dividing its length, return shape `(b,h,len(vals)/(b*h))`, filling each matrix row by row before starting the next matrix.” |
| 928 | D1, D2, D10 | “Given a nonempty rectangular nested list `grid` of shape `(h,w)`, return a flat list containing each complete row in order.” |
| 929 | D1, D2, D7 | “Given a nonempty flat list `temps` containing complete consecutive seven-day weeks, return one maximum per week as a list in week order.” |
| 930 | D1, D8 | “Given a nonempty flat list of floating-point readings `temps` and positive integer `k` dividing its length, return the arithmetic mean of each consecutive block of `k` readings as a list.” |
| 931 | D1, D8 | “Given a nonempty flat list `temps` containing complete consecutive seven-day weeks, return one sum per week as a list in week order.” |
| 932 | D1, D10 | “Given a floating-point nested list `temps` of shape `(weeks,7)` with at least one week, return seven arithmetic means, each combining the same day position across all weeks.” |
| 933 | D1, D2, D8 | “Given a nonempty flat list `temps` containing complete consecutive seven-day weeks, return each reading minus its own week’s maximum, preserving input order and list length.” |
| 934 | D1, D5, D7, D8 | “Given a nonempty flat list of floating-point readings `temps` and positive integer `k` dividing its length, return each reading minus the mean of its consecutive `k`-reading block, preserving input order.” |
| 935 | D1, D9 | “Given a nonempty flat list of floating-point readings `temps` containing complete consecutive seven-day weeks with nonzero means, return each reading divided by its own week’s mean, preserving input order.” |
| 936 | D1, D2, D8 | “Given a floating-point nested list `temps` of shape `(weeks,7)`, return a nested list of the same shape in which each reading has its own row’s arithmetic mean subtracted.” |
| 937 | D1, D2, D9 | “Given a nonempty flat list of floating-point readings `temps` containing complete consecutive seven-day weeks, each with at least two distinct values, return `(reading-week_min)/(week_max-week_min)` for every reading in order, rounded to four decimal places.” |
| 938 | D1, D2, D5, D9, D12 | “Given a flat list of floating-point readings `temps` and integer `k≥2` dividing its length, return each reading’s z-score within its consecutive `k`-reading block, using sample standard deviation and rounding to four decimal places; every block has nonzero standard deviation.” |
| 939 | D1, D2, D9, D12 | “Given a nonempty flat list of floating-point readings `temps` containing complete consecutive seven-day weeks with nonzero sample standard deviations, return each reading divided by its own week’s sample standard deviation, without subtracting the mean, rounded to four decimal places.” |
| 940 | D1, D2, D9, D12 | “Given a floating-point nested list `temps` of shape `(weeks,7)`, return each reading’s z-score within its own row, using sample standard deviation and rounding to four decimal places; every row has nonzero standard deviation.” |

## 2. Lesson pages — per page, per rubric code

### L1 — Worked blocks exceed 16 lines

**`EXPL_INTERLEAVE` — major. All 12 pages.**

Counts below are physical lines inside worked-example fences, excluding fence markers. Each proposed sentence belongs between code blocks; further subdivision remains necessary wherever either block exceeds 16 lines.

| Page | Block length | Exact boundary quote | Prose replacement/insertion |
|---|---:|---|---|
| `kp-pattern-language.md` | 31 | `# Sequence layout swap: batch-first -> time-first.` | “The image example changes where channels appear; next, move the sequence’s time axis ahead of its batch axis while preserving feature order.” |
| `kp-merge-axes.md` | 24 | `# Merge NON-adjacent axes: batch into width -> images side by side.` | “To place complete images side by side, batch must vary more slowly than the columns within each image.” |
| `kp-split-axes.md` | 27 | `# Split a sequence into p-token segments: t = n segments of length p.` | “A known segment length determines how the packed time axis splits into segment number and position within that segment.” |
| `kp-singleton-and-lists.md` | 35 | `# Singleton insertion: a plain 2-D tensor gains a leading axis.` | “Stacking created a batch from several tensors; singleton insertion instead adds a length-one axis to one existing tensor without duplicating its values.” |
| `kp-reduce-model.md` | 34 | `# Keep-a-singleton: per-column max of an image, row axis kept as 1.` | “Keeping a length-one row axis records which dimension was reduced and prepares the column maxima for alignment with the original image.” |
| `kp-repeat-model.md` | 37 | `# 2. STRETCH, factor fast: each ROW repeats consecutively.` | “To keep each row’s copies adjacent, the copy index must vary faster than the source-row index.” |
| `kp-grids-montage.md` | 24 | `# Reverse: carve the montage back into the batch. Each merged input axis` | “Reversing the montage requires one supplied factor for each spatial axis, because neither grid size can be inferred from the montage dimensions alone.” |
| `kp-patches-space-depth.md` | 29 | `# SPACE-TO-DEPTH on a batch: blocks fold into channels; H, W halve.` | “Space-to-depth keeps block locations spatial and places each pixel’s position within its block into the channel axis.” |
| `kp-pooling.md` | 27 | `# Max pooling is the same pattern, different aggregation.` | “The same disjoint windows can produce maxima instead of means; window geometry stays fixed while the aggregation changes.” |
| `kp-dl-flatten-heads.md` | 27 | `# Heads merge: (b, nh, t, d) -> (b, t, (nh d)).` | “For attention output, each token needs one vector containing every head’s features, with all features from head zero before those from head one.” |
| `kp-channel-groups-temporal.md` | 24 | `# Temporal pooling: average adjacent pairs of time steps.` | “The same factor-order rule applies to time: split the sequence into adjacent pairs, then keep the pair number while averaging positions within each pair.” |
| `kp-einsum.md` — repeated-names segment | 17 | `tr = einops.einsum(m, "i i ->")` | “Repeating a name within one operand selects entries whose two coordinates are equal; summing those entries gives the trace.” |

Current worked blocks all print alongside assertions: **`EXPL_PRINTS` passes**. When splitting them, retain a print in each new assertion-bearing block.

### [kp-pattern-language.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-pattern-language.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_CORRECT` — major.** Quote: “One element proves the whole mapping.” One coordinate check cannot establish correctness elsewhere.  
  **Replacement:** “Tracking one element illustrates the intended mapping; checking the complete output against the coordinate rule verifies the whole transformation.”
- **`GIVE_STARTER` — major, faded q345.** `new_syntax: [einops.rearrange]`, but starter retains `return einops.rearrange(arr, '_____')`.  
  **Replacement docstring:** “Return the same images with channels before the spatial dimensions.” Blank both operation and pattern: `einops._____(arr, '_____')`.
- **`EXPL_TRANSFER` — major, q345.** Worked call and faded solution both use `'b h w c -> b c h w'`.  
  **Replacement faded task:** “Given a channels-first image batch of shape `(b,c,h,w)`, return the same images in channels-last layout `(b,h,w,c)`.”
- **`GIVE_RUNG` — major; `DIFF_RUNG` — minor:** integrated 913–916 are single permutations; D10.

### [kp-merge-axes.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-merge-axes.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_TRANSFER` — major, q391.** Example and faded solution both use `'c h w -> c (h w)'`.  
  **Replacement faded task:** “Given an image of shape `(c,h,w)`, place complete channel images vertically in channel order, returning shape `(c*h,w)`.”
- **`GIVE_RUNG` — major; `DIFF_RUNG` — minor:** basic integrated merges identified in D10.

q347 does require a different merge from the flatten example; no transfer finding against that faded task.

### [kp-split-axes.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-split-axes.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_CORRECT` — major.** Quotes: “`'(b w) ... -> b ...'` unpacks” and “`'... -> ... (h p)'` repacks”. As written, first drops named factor `w`; second introduces unsupported names.  
  **Replacement:** “The pattern `'(b w) ... -> b w ...'` exposes both packed factors, while `'... h p -> ... (h p)'` combines two existing axes.”
- **`EXPL_TRANSFER` — major, q390.** Worked example already restores `'c (h w) -> c h w'` with a supplied height. Faded task repeats it.  
  **Replacement faded task:** “Given features of shape `(b,t,heads*d)` and the head count, return shape `(b,t,heads,d)`, keeping tokens before heads.”
- **`GIVE_RUNG` — major; `DIFF_RUNG` — minor:** 921, 924, 925, 928; D10.

### [kp-singleton-and-lists.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-singleton-and-lists.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_INTRO` — major.** Quote: “Task: stack a list of images into a batch; add a singleton channel axis; combine both in one pattern.” Combined example actually stacks and concatenates; it does not combine stacking with singleton insertion.  
  **Replacement:** “We will stack an image list, insert a singleton into a separate tensor, concatenate another image list, and check an invalid squeeze.”
- **`EXPL_TRANSFER` — major, q361.** Example and faded solution both use `'b h w c -> b h w c'` on an image list.  
  **Replacement faded task:** “Given a list of channels-first tensors of shape `(c,h,w)`, return a channels-last batch of shape `(b,h,w,c)`, preserving list order.”

### [kp-reduce-model.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-reduce-model.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_CORRECT` — major.** Quote: “reduce requires it (something must reduce).” A reduction call can retain every axis; identity reduction was accepted in local einops mathematical check.  
  **Replacement:** “Unlike rearrange, reduce permits axes to disappear and specifies how to aggregate them; if every axis survives unchanged, no values are combined.”
- **`GIVE_STARTER` — major, q325.** `new_syntax: [einops.reduce]`; starter retains `einops.reduce(arr, '_____', 'mean')`.  
  **Replacement docstring:** “Return one arithmetic mean per image, averaging its channels and pixels.” Blank the newly introduced function name as well as pattern.
- **`EXPL_TRANSFER` — major, q325.** Worked example already computes `'b c h w -> b'` using `'mean'`.  
  **Replacement faded task:** “Given a batch of shape `(b,c,h,w)`, return one mean per channel across every image and spatial position, with shape `(c,)`.”

### [kp-repeat-model.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-repeat-model.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_CORRECT` — major.** Quotes: “The new-axis case allocates real copies (unlike `expand`'s virtual stretch)” and “repeat materializes an actual tensor with the new shape”. Allocation is not guaranteed; installed einops PyTorch backend implements added axes through `expand`.  
  **Replacement:** “Repeat specifies repeated values and an output shape; the backend may return a view or allocate storage, so do not infer memory independence from the operation’s name.”
- **`GIVE_STARTER` — major, q317.** `new_syntax: [einops.repeat]`; starter retains `einops.repeat(img, 'c h w -> c (_____) w')`.  
  **Replacement docstring:** “Return an image whose rows each appear three times consecutively.” Blank `repeat` while retaining the structural scaffold.
- **`EXPL_PREREQ` — major, 938–940.** Page owns custom-standard-deviation tasks but teaches no custom reduction interface; D12.
- **`GIVE_EXAMPLE` — major, 903:** exact worked-input/output reuse; D13.
- **`GIVE_RUNG` — major; `DIFF_RUNG` — minor:** simple repeat tasks identified in D10.

### [kp-grids-montage.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-grids-montage.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_TRANSFER` — major, q389.** Worked intro: “six images into a 3-row × 2-column grid, row-major”. Faded task: “Six images → 3×2 grid, row-major.” Same factors, placement, and pattern.  
  **Replacement faded task:** “Arrange six images in a three-row, two-column grid, filling each column from top to bottom before moving right.”
- **`EXPL_CORRECT` — major.** Quote: “It's rows because it's the SLOW factor of the batch split AND merges with h.” A row coordinate can be the fast input factor in column-major filling.  
  **Replacement:** “The factor is the grid-row coordinate because the output combines it with image height; its input position determines whether image order fills rows or columns first.”

### [kp-patches-space-depth.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-patches-space-depth.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_CORRECT` — major.** Quote: “the block coordinates fold into the CHANNEL axis”. Pattern retains block coordinates `h,w` spatially and folds within-block offsets `p,q` into channels.  
  **Replacement:** “Space-to-depth keeps each block’s grid location on the spatial axes and folds the within-block row and column offsets into the channel axis.”
- **`EXPL_CORRECT` — major.** Quote: “it was your grid-KP faded exercise (q323)”. Frontmatter assigns q323 to this page; grid page’s faded ID is 389.  
  **Replacement:** “Reassembly reverses patch extraction; this page practices that direction in q323.”
- **`EXPL_TRANSFER` — major, q323.** Worked example already reassembles with `'(h w) p1 p2 c -> (h p1) (w p2) c'`; faded solution repeats it.  
  **Replacement faded task:** “Given a row-major stack of channels-first patches of shape `(grid_rows*grid_cols,c,p1,p2)`, reconstruct one channels-first image of shape `(c,grid_rows*p1,grid_cols*p2)`.”

### [kp-pooling.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-pooling.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_TRANSFER` — major, q324.** Both worked task and faded task perform “2×2 non-overlapping average pooling” on a channels-first batch.  
  **Replacement faded task:** “Given a channels-first image batch whose height is divisible by three, average each consecutive group of three rows while preserving batch, channels, and width.”
- **`EXPL_CORRECT` — major.** Quote: “Overlapping / strided pooling is outside reduce's power”. Disjoint pooling already has a stride equal to window size.  
  **Replacement:** “A factored reduction supports disjoint windows whose stride equals their size; overlapping windows require another window-extraction operation.”
- **`STMT_CASES` — major:** owned drills 917, 918, 920 have insufficient mapping coverage; D6.

### [kp-dl-flatten-heads.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-dl-flatten-heads.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_CORRECT` — major.** Quote: “TOTAL flattening to a scalar count of axes”. Flattening produces a one-dimensional collection of entries, not a scalar count.  
  **Replacement:** “Flattening the whole tensor produces a one-dimensional tensor containing every entry; its length is the product of the original axis lengths.”
- **`EXPL_TRANSFER` — major, q356.** Worked and faded solutions both use `'b c h w -> b (c h w)'`.  
  **Replacement faded task:** “Given one image of shape `(c,h,w)`, return a batch containing its flattened feature vector, with shape `(1,c*h*w)` and channel-major, row-major ordering.”

### [kp-channel-groups-temporal.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-channel-groups-temporal.md) — major

- **`EXPL_INTERLEAVE` — major:** L1.
- **`EXPL_INTRO` — major.** Quote: “split interleaved channel groups; swap group blocks; average-pool time pairs.” Example compares two splits but never performs a group-block swap.  
  **Replacement:** “We will compare interleaved and contiguous interpretations of the same channel axis, then average adjacent pairs of time steps.”
- **`EXPL_CORRECT` — major.** Quote: “channel axis interleaves `coord` groups of k, group index SLOWEST”. Conflicts with page’s own definition of slow group index as contiguous groups.  
  **Replacement:** “The channel axis contains contiguous groups of `k` channels, with the group index varying slowest.”
- **`EXPL_TRANSFER` — major, q362.** Worked example explicitly solves interleaved group extraction with `'b (c g) h w -> g b c h w'`; faded task asks same extraction.  
  **Replacement faded task:** “Reorder contiguous channel groups into an interleaved channel axis, preserving tensor shape `(b,g*c,h,w)` and the order of channels within each group.”

### [kp-einsum.md](/home/stellar-thread/Applications/Delta-Drills-Local/Local_Deployed_Shared/lessons/einops/kp-einsum.md) — major

- **`EXPL_INTERLEAVE` — major:** repeated-names segment, L1.
- **`EXPL_INTRO` — major, all four segments.** Repeated exact structure:

  ````markdown
  ## Worked example

  ```python
  ````

  No introductory prose. Replacements, respectively:

  1. “We will use one matrix to compare its total, column sums, row sums, and transpose, checking how the surviving axes determine each output.”
  2. “We will multiply a matrix by a vector and then by another matrix, checking which shared axis is summed.”
  3. “We will compare corresponding-entry products, all-pairs products, and diagonal selection using small vectors and a square matrix.”
  4. “We will perform separate products for each batch entry, then compare this with one dot product per row.”

- **`EXPL_WHY` — major, repeated-names segment.** Quote: “the SAME two rules decide every case — count which names are shared (those multiply) and which are missing on the right (those sum).” This explanation omits the within-operand equality constraint needed to predict diagonal selection.  
  **Replacement:** “Shared names across operands align factors for multiplication; repeated names within one operand restrict its coordinates to equal indices, and names omitted from the output are then summed.”
- **`GIVE_STARTER` — major, q847–851.** Page scaffolds blank the function but retain docstrings such as “Column sums via einsum.” and “The diagonal via a repeated index.” Apply D3 replacements.
- **`GIVE_PROMPT` — major, faded task hints.** Quotes and replacements:

  | ID | Exact quote | Replacement |
  |---|---|---|
  | 847 | “Which axis disappears when you keep only `j`?” | “Return one total per matrix column.” |
  | 849 | “The shared axis is the vector's ONLY axis, and it is the matrix's first one this time.” | “Return the weighted sum of the matrix rows, using the vector entries as row weights.” |
  | 850 | “Name it the same, and the transpose is free.” | “Return every pairwise dot product between a row of the first matrix and a row of the second.” |
  | 851 | “The trace repeats a name and keeps nothing. Keep it instead.” | “Return the main-diagonal entries in order.” |
  | 852 | “Both names shared, both dropped.” | “Return the sum of all corresponding-entry products.” |
  | 853 | “with a `b` carried through both operands and the output.” | “Pair each matrix with the vector at the same batch position.” |
  | 854 | “Outer product shares no name — except the batch, which is carried through.” | “Return one outer-product matrix for each corresponding pair of batch vectors.” |

- **`GIVE_EXAMPLE`, `EXPL_TRANSFER` — major, q847.** Worked column-sum example is the drill’s first graded case and same operation; D13.
- **`GIVE_RUNG` — major; `DIFF_RUNG` — minor:** basic integrated operations identified in D10.
- **`DIFF_DUP` — minor:** integrated 863 and 875; D11.

Across all pages, **`EXPL_GENERAL_FIRST` passes**: each concept section states a general rule before its worked example. No glossary-order or prerequisite finding asserted merely from terminology whose earlier definition was outside supplied scope.

## 3. Systemic patterns

| Pattern | Rubric codes | Impact |
|---|---|---|
| 72 prompts begin with invisible exercise references | `STMT_SELF_CONTAINED` | Standalone delivery breaks inherited context; 910 and weekly-reading tasks lose material contracts. |
| Prompt instructions and starter docstrings name assessed operations | `GIVE_PROMPT`, `GIVE_STARTER` | Learner receives function choice or derivation before solving. |
| Eight faded-owned bank rows contain from-scratch starters | `GIVE_RUNG` | Bank and lesson scaffolds disagree; fix source consistency before relying on rung labels. |
| 21 case sets lack an edge; three pooling sets miss important mappings | `STMT_CASES`, `CORR_DECISIVE` | Different expected values alone do not establish decisive coverage. |
| Four near-miss explanations misdescribe behavior | `STMT_NEAR_MISS` | Feedback teaches incorrect failure diagnosis. |
| Simple operations populate integrated slots | `GIVE_RUNG`, `DIFF_RUNG`, `DIFF_DUP` | Promotion can reward repetition of already-demonstrated moves. |
| Worked fences are oversized on every page | `EXPL_INTERLEAVE` | Explanations arrive after long blocks instead of alongside decisions. |
| Numerous faded tasks repeat the worked transformation | `EXPL_TRANSFER`; sometimes `GIVE_EXAMPLE` | Learner can reproduce the demonstrated pattern without transferring it. |
| Normalization tasks omit valid denominator domains and callable-reduction teaching | `STMT_INPUT`, `STMT_TERMS`, `EXPL_PREREQ` | Valid inputs, mathematical convention, and required syntax remain under-specified. |
| Several lessons confuse layout semantics with proof, storage, or coordinate roles | `EXPL_CORRECT` | Correct numerical examples coexist with false general explanations. |

Positive findings:

- All 94 IDs have exactly one owning KP among supplied pages.
- Every case set contains differing expected outputs; a single constant cannot satisfy all cases.
- Returning the first input unchanged fails at least one case for every drill.
- Stored wrong-example outputs differ from their corresponding first expected outputs; 917’s explanation is the exception in *described behavior*, not literal stored output.
- Image drills **889–920** express transformations using named axes. No `CORR_VISUAL` defect established from these fields.
- No `CORR_TYPES` failure established: stored list/scalar structures align with answers. Actual floating-point comparison policy remains unverified.

## 4. Top 10 fixes with rewrites

Ranking prioritizes incorrect solutions surviving grading, contradictory contracts, undefined numerical domains, then standalone usability and answer exposure. Rewrites below are complete prompt cores with concrete examples; scaffold/rung/case repairs remain necessary where identified.

### 1. q917 — Described near miss passes every case

**`CORR_DECISIVE`, `STMT_CASES` — major; `STMT_NEAR_MISS` — minor.**

> Given a channels-first image `img` as a nested list of shape `(c,h,w)`, replace each disjoint 2×2 block in each channel with its maximum. Return a nested list of shape `(c,h/2,w/2)`.

- Constraints: positive dimensions; even `h` and `w`; windows start at the top-left corner and cover the image.
- Example: `img=[[[1,2],[3,4],[9,8],[7,6]]]` → `[[[4],[9]]]`.
- Explanation: upper block’s maximum is `4`; lower block’s maximum is `9`.

Add this height-four case to grading; repair near-miss explanation.

### 2. q920 — Concatenation never distinguished from interleaving

**`STMT_CASES`, `STMT_INPUT`, `STMT_SELF_CONTAINED` — major.**

> Given a nested list `imgs` of shape `(b,c,h,w)`, replace each disjoint 2×2 block with its maximum, then place the resulting images side by side in batch order. Return one channels-first image as a nested list of shape `(c,h/2,b*(w/2))`.

- Constraints: positive dimensions; even `h` and `w`; preserve every complete image before starting the next along output width.
- Example: `imgs=[[[[1,2,5,6],[3,4,7,8]]],[[[10,20,50,60],[30,40,70,80]]]]` → `[[[4,8,40,80]]]`.
- Explanation: resulting image rows `[4,8]` and `[40,80]` remain contiguous.

### 3. q936 — Grader violates seven-day input contract

**`STMT_INPUT`, `STMT_CASES`, `GIVE_PROMPT`, `STMT_SELF_CONTAINED` — major.**

> Given floating-point readings `temps` as a nested list of shape `(weeks,7)`, subtract each week’s arithmetic mean from every reading in that week. Return a nested list of the same shape, preserving week and day order.

- Constraints: at least one week; exactly seven readings per row.
- Example: `temps=[[1,2,3,4,5,6,7]]` → `[[-3,-2,-1,0,1,2,3]]`.
- Explanation: this week’s mean is `4`.

Replace the two-day graded row with a valid seven-day edge case.

### 4. q938 — Implementation supplied; statistical domain and prerequisite missing

**`GIVE_PROMPT`, `STMT_INPUT`, `EXPL_PREREQ`, `STMT_CASES`, `STMT_SELF_CONTAINED` — major.**

> Given a flat list of floating-point readings `temps` and a block length `k`, return each reading’s z-score within its consecutive block: the reading minus its block mean, divided by its block’s sample standard deviation. Preserve input order and round each result to four decimal places.

- Constraints: `k≥2`; nonempty length divisible by `k`; each block has nonzero standard deviation.
- Definition: sample variance divides the sum of squared deviations by `k-1`.
- Example: `temps=[1.0,3.0]`, `k=2` → `[-0.7071,0.7071]`.

Teach custom reductions in the lesson; remove the prescribed call from prompt.

### 5. q937 — Constant-week behavior unspecified

**`STMT_INPUT`, `GIVE_PROMPT`, `STMT_SELF_CONTAINED` — major.**

> Given a flat list of floating-point readings `temps` containing consecutive seven-day weeks, return each reading normalized within its own week as `(reading-week_min)/(week_max-week_min)`. Preserve input order and round each result to four decimal places.

- Constraints: nonempty length divisible by seven; every week contains at least two distinct readings.
- Example: `temps=[1,2,3,4,5,6,7]` → `[0,0.1667,0.3333,0.5,0.6667,0.8333,1]`.

This explicitly excludes the division-by-zero case unsupported by current answer.

### 6. q935 — Input grouping and nonzero denominator missing

**`STMT_INPUT`, `STMT_SELF_CONTAINED` — major.**

> Given a flat list of floating-point readings `temps` containing consecutive seven-day weeks, divide each reading by its own week’s arithmetic mean. Return a list of the same length in the original order.

- Constraints: nonempty length divisible by seven; every week’s mean is nonzero.
- Example: `temps=[1,2,3,4,5,6,7]` → `[0.25,0.5,0.75,1,1.25,1.5,1.75]`.
- Explanation: all seven readings belong to a week whose mean is `4`.

### 7. q910 — “Same grid” conceals essential contract

**`STMT_SELF_CONTAINED`, `STMT_INPUT` — major; `STMT_NEAR_MISS` — minor.**

> Given channels-first images `imgs` as a nested list of shape `(b,c,h,w)` and a column count `cols`, arrange complete images in batch order, filling each grid row from left to right before starting the next row. Return a nested list of shape `(c,(b/cols)*h,cols*w)`.

- Constraints: positive dimensions; positive integer `cols` divides `b`.
- Example: `imgs=[[[[1]]],[[[2]]],[[[3]]],[[[4]]],[[[5]]],[[[6]]]]`, `cols=3` → `[[[1,2,3],[4,5,6]]]`.

Use rectangular grid examples to distinguish row count from column count; correct the existing near-miss explanation.

### 8. q847 — Copied worked answer plus wrong faded starter

**`GIVE_EXAMPLE`, `GIVE_STARTER`, `GIVE_PROMPT`, `GIVE_RUNG`, `STMT_CASES` — major.**

For a transfer-bearing faded replacement:

> Given a nested list `x` of shape `(b,r,c)`, return the column sums of each matrix independently. Return a nested list of shape `(b,c)`, preserving batch and column order.

- Constraints: positive dimensions; rectangular matrices of equal shape.
- Example: `x=[[[2,-1],[4,3]],[[5,6],[0,-2]]]` → `[[6,2],[5,4]]`.
- Explanation: each output row summarizes only the corresponding input matrix.

Update function signature, scaffold, and cases together. Leave operation name and pattern blank.

### 9. q918 — No case tests multiple height windows

**`STMT_CASES`, `STMT_INPUT`, `STMT_SELF_CONTAINED` — major.**

> Given a floating-point channels-first image `img` as a nested list of shape `(c,h,w)`, replace each disjoint 2×2 block in each channel with its arithmetic mean. Return a nested list of shape `(c,h/2,w/2)`.

- Constraints: positive dimensions; even `h` and `w`; no padding or overlapping windows.
- Example: `img=[[[1,2],[3,4],[9,8],[7,6]]]` → `[[[2.5],[7.5]]]`.
- Explanation: the upper four values average to `2.5`; the lower four average to `7.5`.

Grade this example in addition to existing width-oriented cases.

### 10. q939 — Standard-deviation convention and valid domain hidden behind function name

**`STMT_INPUT`, `GIVE_PROMPT`, `EXPL_PREREQ`, `STMT_SELF_CONTAINED` — major.**

> Given a flat list of floating-point readings `temps` containing consecutive seven-day weeks, divide each reading by its own week’s sample standard deviation without subtracting the mean. Preserve input order and round each result to four decimal places.

- Constraints: nonempty length divisible by seven; every week has nonzero standard deviation.
- Definition: each week’s sample variance divides the sum of squared deviations from its mean by six.
- Example: `temps=[1,2,3,4,5,6,7]` → `[0.4629,0.9258,1.3887,1.8516,2.3146,2.7775,3.2404]`.

---

Git: MODERATE — repo=Delta-Drills-Local branch=main; staged=0, unstaged=3, untracked=6, conflicts=0. Existing mixed scopes preserved; no files edited.