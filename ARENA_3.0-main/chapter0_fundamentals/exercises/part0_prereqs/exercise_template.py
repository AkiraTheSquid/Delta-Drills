import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def setup_environment() -> dict:
    """Configure imports and paths so this file can run standalone."""
    chapter = "chapter0_fundamentals"
    section = "part0_prereqs"

    candidates = [Path.cwd(), *Path.cwd().parents]
    repo_dir = next((p / "ARENA_3.0-main" for p in candidates if (p / "ARENA_3.0-main").exists()), None)
    if repo_dir is not None and (repo_dir / chapter).exists():
        root_dir = repo_dir
    else:
        root_dir = next((p for p in candidates if (p / chapter).exists()), Path.cwd())

    exercises_dir = root_dir / chapter / "exercises"
    section_dir = exercises_dir / section

    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

    import einops  # noqa: F401
    import torch as t  # noqa: F401
    from torch import Tensor  # noqa: F401

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

import einops
import torch as t
from torch import Tensor

try:
    import part0_prereqs.tests as tests
    from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img
except Exception:
    tests = None  # type: ignore[assignment]
    display_array_as_img = None  # type: ignore[assignment]
    display_soln_array_as_img = None  # type: ignore[assignment]


def load_numbers(path_override: Optional[Path] = None) -> np.ndarray:
    """Load numbers.npy from the exercise folder."""
    candidates = []
    if path_override is not None:
        candidates.append(Path(path_override))
    candidates.append(PATHS["section_dir"] / "numbers.npy")
    candidates.append(PATHS["root_dir"] / "einops" / "numbers.npy")

    for path in candidates:
        if path.exists():
            return np.load(path)

    raise FileNotFoundError("numbers.npy not found. Checked: " + ", ".join(str(path) for path in candidates))


def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol: float = 1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")


def save_numpy_image(img_array: np.ndarray, out_path: Path) -> None:
    """Save an image array to disk when notebook rendering is unavailable."""
    try:
        from PIL import Image
    except Exception:
        print("Install pillow to save images: pip install pillow")
        return

    if img_array.ndim == 3:
        if img_array.shape[0] in (1, 3, 4):
            img = np.moveaxis(img_array, 0, -1)
        else:
            raise ValueError(f"Unexpected channel dimension: {img_array.shape}")
    elif img_array.ndim == 2:
        img = img_array
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.astype(np.uint8, copy=False)).save(out_path)
    print("Saved image:", out_path)


# =========================
# Optional image practice
# =========================
#
# Uncomment when you want to practice the image-display section from the notebook.
#
# arr = load_numbers()
#
# arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
# display_array_as_img(arr1)
# display_soln_array_as_img(1)
#
# arr2 = ...
# display_array_as_img(arr2)
# display_soln_array_as_img(2)


# =========================
# Place your solutions below
# =========================


def rearrange_1() -> Tensor:
    """Return [[3, 4], [5, 6], [7, 8]] using only t.arange and einops.rearrange."""
    raise NotImplementedError()


def rearrange_2() -> Tensor:
    """Return [[1, 2, 3], [4, 5, 6]] using only t.arange and einops.rearrange."""
    raise NotImplementedError()


def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week."""
    raise NotImplementedError()


def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to."""
    raise NotImplementedError()


def temperatures_normalized(temps: Tensor) -> Tensor:
    """Normalize each day by the weekly average and weekly standard deviation."""
    raise NotImplementedError()


def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row by its L2 norm."""
    raise NotImplementedError()


def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for the rows."""
    raise NotImplementedError()


def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Draw n samples from a discrete distribution over class probabilities probs."""
    raise NotImplementedError()


def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of correct argmax predictions."""
    raise NotImplementedError()


def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Return the total price of items using direct indexing."""
    raise NotImplementedError()


def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Return matrix values gathered row-wise from the given column indexes."""
    raise NotImplementedError()


def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Return the total price of items using gather."""
    raise NotImplementedError()


def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return matrix values at the coordinates in coords."""
    raise NotImplementedError()


def batched_logsumexp(matrix: Tensor) -> Tensor:
    """Stable logsumexp across the last dimension."""
    raise NotImplementedError()


def batched_softmax(matrix: Tensor) -> Tensor:
    """Softmax across the last dimension."""
    raise NotImplementedError()


def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Stable log-softmax across the last dimension."""
    raise NotImplementedError()


def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Return the per-example cross entropy loss."""
    raise NotImplementedError()


def collect_rows(matrix: Tensor, row_indexes: Tensor) -> Tensor:
    """Collect rows from matrix according to row_indexes."""
    raise NotImplementedError()


def collect_columns(matrix: Tensor, column_indexes: Tensor) -> Tensor:
    """Collect columns from matrix according to column_indexes."""
    raise NotImplementedError()


def einsum_trace(mat: np.ndarray):
    """Return the trace using einops.einsum."""
    raise NotImplementedError()


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """Matrix-vector multiplication using einops.einsum."""
    raise NotImplementedError()


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """Matrix-matrix multiplication using einops.einsum."""
    raise NotImplementedError()


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """Inner product using einops.einsum."""
    raise NotImplementedError()


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """Outer product using einops.einsum."""
    raise NotImplementedError()


if __name__ == "__main__":
    print("exercise template ready")
    print("root_dir:", PATHS["root_dir"])
    print("section_dir:", PATHS["section_dir"])
    print("numbers.npy exists:", (PATHS["section_dir"] / "numbers.npy").exists())
    print("Fill in the function bodies above, then call the checks you want to run.")
    print("Example:")
    print("  tests.test_einsum_trace(einsum_trace)")
    print("  assert_all_equal(rearrange_1(), t.tensor([[3, 4], [5, 6], [7, 8]]))")
