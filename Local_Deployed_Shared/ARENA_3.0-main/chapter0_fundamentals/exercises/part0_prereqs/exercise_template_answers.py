import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def setup_environment() -> dict:
    """Configure imports and important paths so solutions can run standalone in VS Code.

    Returns a dict containing useful paths: root_dir, exercises_dir, section_dir.
    """
    chapter = "chapter0_fundamentals"
    section = "part0_prereqs"

    # Locate repository root robustly
    candidates = [Path.cwd(), *Path.cwd().parents]
    repo_dir = next((p / "ARENA_3.0-main" for p in candidates if (p / "ARENA_3.0-main").exists()), None)
    if repo_dir is not None and (repo_dir / chapter).exists():
        root_dir = repo_dir
    else:
        # Fallback: nearest ancestor that directly contains the chapter folder
        root_dir = next((p for p in candidates if (p / chapter).exists()), Path.cwd())

    exercises_dir = root_dir / chapter / "exercises"
    section_dir = exercises_dir / section

    # Ensure local imports like `import part0_prereqs.tests as tests` work
    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

    # Import libraries after sys.path is set
    import einops  # noqa: F401
    import torch as t  # noqa: F401
    from torch import Tensor  # noqa: F401

    # Optionally make working directory the exercises dir (useful for relative assets)
    try:
        os.chdir(str(exercises_dir))
    except Exception:
        pass

    return {
        "root_dir": root_dir,
        "exercises_dir": exercises_dir,
        "section_dir": section_dir,
    }


PATHS = setup_environment()

# After environment is ready, import course utilities and tests
import einops
import torch as t
from torch import Tensor

try:
    import part0_prereqs.tests as tests
    from part0_prereqs.utils import (
        display_array_as_img,
        display_soln_array_as_img,
    )
except Exception:
    tests = None  # type: ignore[assignment]
    display_array_as_img = None  # type: ignore[assignment]
    display_soln_array_as_img = None  # type: ignore[assignment]


def load_numbers(path_override: Optional[Path] = None) -> np.ndarray:
    """Load numbers.npy, trying common locations.

    Order:
    1) explicit path_override
    2) <section_dir>/numbers.npy
    3) <root_dir>/einops/numbers.npy
    """
    candidates = []
    if path_override is not None:
        candidates.append(Path(path_override))
    candidates.append(PATHS["section_dir"] / "numbers.npy")
    candidates.append(PATHS["root_dir"] / "einops" / "numbers.npy")

    for p in candidates:
        if p.exists():
            return np.load(p)

    raise FileNotFoundError(
        "numbers.npy not found. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


    # -----------------------------
    # Function checks (quick self-tests)
    # -----------------------------
    def _run_checks() -> None:
        print("\nRunning function checks...")

        # A-section
        assert_all_equal(rearrange_1(), t.tensor([[3, 4], [5, 6], [7, 8]]))
        assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))

        # B-section
        temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
        assert_all_close(temperatures_average(temps), t.tensor([71.571, 79.0]))
        expected_b2 = t.tensor([-0.571, 0.429, -1.571, 3.429, -0.571, 0.429, -1.571, -4.0, 1.0, 6.0, 1.0, -1.0, -7.0, 4.0])
        assert_all_close(temperatures_differences(temps), expected_b2)
        expected_b3 = t.tensor([-0.333, 0.249, -0.915, 1.995, -0.333, 0.249, -0.915, -0.894, 0.224, 1.342, 0.224, -0.224, -1.565, 0.894])
        assert_all_close(temperatures_normalized(temps), expected_b3)

        # C-section
        matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
        expected_norm = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
        assert_all_close(normalize_rows(matrix), expected_norm)
        expected_cos = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
        assert_all_close(cos_sim_matrix(matrix), expected_cos)

        # D-section (use moderate n for speed)
        n = 200_000
        probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
        freqs = t.bincount(sample_distribution(probs, n)) / n
        assert_all_close(freqs, probs, atol=5e-3)

        # E-section
        scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
        true_classes = t.tensor([0, 1, 0])
        assert classifier_accuracy(scores, true_classes) == 2.0 / 3.0

        # F-section
        prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
        items = t.tensor([0, 0, 1, 1, 4, 3, 2])
        assert total_price_indexing(prices, items) == 9.0
        assert total_price_gather(prices, items) == 9.0

        mat = t.arange(15).view(3, 5)
        idx = t.tensor([[4], [3], [2]])
        assert_all_equal(gather_2d(mat, idx), t.tensor([[4], [8], [12]]))
        idx2 = t.tensor([[2, 4], [1, 3], [0, 2]])
        assert_all_equal(gather_2d(mat, idx2), t.tensor([[2, 4], [6, 8], [10, 12]]))

        # G-section
        mat_2d = t.arange(15).view(3, 5)
        coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
        assert_all_equal(integer_array_indexing(mat_2d, coords_2d), t.tensor([1, 4, 9]))
        mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
        coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
        assert_all_equal(integer_array_indexing(mat_3d, coords_3d), t.tensor([0, 5, 10, 15, 20]))

        # H-section
        m1 = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]]).float()
        expected_m1 = t.tensor([-1000 + np.log(4), 1000 + np.log(4)])
        assert_all_close(batched_logsumexp(m1), expected_m1)
        m2 = t.randn((10, 20))
        assert_all_close(batched_logsumexp(m2), t.logsumexp(m2, dim=-1))

        m3 = t.arange(1, 6).view((1, 5)).float().log()
        expected_soft = t.arange(1, 6).view((1, 5)) / 15.0
        assert_all_close(batched_softmax(m3), expected_soft)
        m3b = t.rand((10, 20))
        soft = batched_softmax(m3b)
        assert soft.min() >= 0.0 and soft.max() <= 1.0
        assert_all_equal(soft.argsort(), m3b.argsort())
        assert_all_close(soft.sum(dim=-1), t.ones(m3b.shape[:-1]))

        start = 1000
        m4 = t.arange(start + 1, start + 7).view((2, 3)).float()
        expected_logsoft = t.tensor([[-2.4076, -1.4076, -0.4076], [-2.4076, -1.4076, -0.4076]])
        assert_all_close(batched_logsoftmax(m4), expected_logsoft)

        logits = t.tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
        labels = t.tensor([2, 0, 0])
        expected_ce = t.tensor([0.0, np.log(3), float("inf")])
        assert_all_close(batched_cross_entropy_loss(logits, labels), expected_ce)

        # I-section
        m5 = t.arange(15).view((5, 3))
        rows = t.tensor([0, 2, 1, 0])
        assert_all_equal(collect_rows(m5, rows), t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]]))
        cols = t.tensor([0, 2, 1, 0])
        assert_all_equal(collect_columns(m5, cols), t.tensor([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]]))

        # Einsum
        A = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        C = np.array([[1, 2, 3], [4, 5, 6]])
        D = np.array([[7, 8], [9, 10], [11, 12]])
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])

        assert einsum_trace(A) == np.trace(A)
        np.testing.assert_allclose(einsum_mv(A, b), A @ b)
        np.testing.assert_allclose(einsum_mm(C, D), C @ D)
        assert einsum_inner(v1, v2) == np.inner(v1, v2)
        np.testing.assert_allclose(einsum_outer(v1, v2), np.outer(v1, v2))

        print("All checks passed ✔")

    _run_checks()
# Lightweight test helpers (mirrors the notebook utilities)
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"


def assert_all_close(actual: Tensor, expected: Tensor, atol: float = 1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)


# Utility: save numpy image to PNG (fallback for non-notebook runs)
def save_numpy_image(img_array: np.ndarray, out_path: Path) -> None:
    try:
        from PIL import Image
    except Exception:
        print("Install pillow to save images: pip install pillow")
        return

    if img_array.ndim == 3:
        # Expecting (C, H, W) -> convert to (H, W, C)
        if img_array.shape[0] in (1, 3, 4):
            img = np.moveaxis(img_array, 0, -1)
        else:
            raise ValueError(f"Unexpected channel dimension: {img_array.shape}")
    elif img_array.ndim == 2:
        img = img_array
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    img_uint8 = img.astype(np.uint8, copy=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8).save(out_path)
    print("Saved image:", out_path)


# =========================
# Place your solutions below
# =========================

# A-section (einops.rearrange basics)
def rearrange_1() -> Tensor:
    return einops.rearrange(t.arange(3, 9), "(h w) -> h w", h=3, w=2)


def rearrange_2() -> Tensor:
    return einops.rearrange(t.arange(1, 7), "(h w) -> h w", h=2, w=3)


# B-section (reduce, broadcasting)
def temperatures_average(temps: Tensor) -> Tensor:
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(h 7) -> h", "mean")


def temperatures_differences(temps: Tensor) -> Tensor:
    assert len(temps) % 7 == 0
    avg = einops.reduce(temps, "(w 7) -> w", "mean")
    return temps - einops.repeat(avg, "w -> (w 7)")


def temperatures_normalized(temps: Tensor) -> Tensor:
    avg = einops.reduce(temps, "(w 7) -> w", "mean")
    std = einops.reduce(temps, "(h 7) -> h", t.std)
    return (temps - einops.repeat(avg, "w -> (w 7)")) / einops.repeat(std, "w -> (w 7)")


# C-section (normalize and cosine similarity)
def normalize_rows(matrix: Tensor) -> Tensor:
    row_norms = matrix.norm(dim=1, keepdim=True)
    return matrix / row_norms


def cos_sim_matrix(matrix: Tensor) -> Tensor:
    matrix_normalized = normalize_rows(matrix)
    return matrix_normalized @ matrix_normalized.T


# D-section (sampling)
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    assert abs(probs.sum() - 1.0) < 0.001
    assert (probs >= 0).all()
    return (t.rand(n, 1) > t.cumsum(probs, dim=0)).sum(dim=-1)


# E-section (classifier accuracy)
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    assert true_classes.max() < scores.shape[1]
    return (scores.argmax(dim=1) == true_classes).float().mean()


# F-section (indexing & gather)
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    assert items.max() < prices.shape[0]
    return prices[items].sum().item()


def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    assert matrix.ndim == indexes.ndim
    assert indexes.shape[0] <= matrix.shape[0]
    out = matrix.gather(1, indexes)
    assert out.shape == indexes.shape
    return out


def total_price_gather(prices: Tensor, items: Tensor) -> float:
    assert items.max() < prices.shape[0]
    return prices.gather(0, items).sum().item()


# G-section (integer array indexing)
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    return matrix[tuple(coords.T)]


# H-section (stability, softmax family, loss)
def batched_logsumexp(matrix: Tensor) -> Tensor:
    C = matrix.max(dim=-1).values
    exps = t.exp(matrix - einops.rearrange(C, "n -> n 1"))
    return C + t.log(t.sum(exps, dim=-1))


def batched_softmax(matrix: Tensor) -> Tensor:
    exp = matrix.exp()
    return exp / exp.sum(dim=-1, keepdim=True)


def batched_logsoftmax(matrix: Tensor) -> Tensor:
    C = matrix.max(dim=1, keepdim=True).values
    return matrix - C - (matrix - C).exp().sum(dim=1, keepdim=True).log()


def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]
    logprobs = batched_logsoftmax(logits)
    indices = einops.rearrange(true_labels, "n -> n 1")
    pred_at_index = logprobs.gather(1, indices)
    return -einops.rearrange(pred_at_index, "n 1 -> n")


# I-section (collect rows/cols)
def collect_rows(matrix: Tensor, row_indexes: Tensor) -> Tensor:
    assert row_indexes.max() < matrix.shape[0]
    return matrix[row_indexes]


def collect_columns(matrix: Tensor, column_indexes: Tensor) -> Tensor:
    assert column_indexes.max() < matrix.shape[1]
    return matrix[:, column_indexes]


# Einsum section (numpy + einops)
def einsum_trace(mat: np.ndarray):
    return einops.einsum(mat, "i i ->")


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    return einops.einsum(mat, vec, "i j, j -> i")


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    return einops.einsum(mat1, mat2, "i k, k j -> i j")


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    return einops.einsum(vec1, vec2, "i, i ->")


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    return einops.einsum(vec1, vec2, "i, j -> i j")


if __name__ == "__main__":
    # Example: quickly verify environment and data access
    print("exercise template ready")
    print("root_dir:", PATHS["root_dir"])  # e.g., .../ARENA_3.0-main
    print("section_dir:", PATHS["section_dir"])  # e.g., .../part0_prereqs
    numbers_path = PATHS["section_dir"] / "numbers.npy"
    print("numbers.npy exists:", numbers_path.exists())
    # Demo: load numbers and visualize a stacked image row (if utils available)
    try:
        arr = load_numbers()
        # Vertical stacking of digits: stack along height
        arr_stacked = einops.rearrange(arr, "b c h w -> c (b h) w")
        print("arr_stacked shape:", arr_stacked.shape)
        if display_array_as_img is not None:
            display_array_as_img(arr_stacked)
        else:
            out_png = PATHS["section_dir"] / "numbers_stacked.png"
            save_numpy_image(arr_stacked, out_png)
            print("Open the saved image to verify.")
    except Exception as e:
        print("Demo failed:", e)
        print(
            "Tip: if you see FileNotFoundError, either place numbers.npy in",
            PATHS["section_dir"],
            "or in",
            PATHS["root_dir"] / "einops",
        )