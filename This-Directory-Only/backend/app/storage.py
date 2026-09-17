from pathlib import Path

from app.config import settings


def ensure_storage_root() -> Path:
    root = Path(settings.storage_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_root(job_id: str) -> Path:
    root = ensure_storage_root() / "jobs" / job_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def job_input_path(job_id: str, original_filename: str) -> Path:
    safe_name = original_filename.replace("..", "_").replace("/", "_").replace("\\", "_")
    return job_root(job_id) / f"input_{safe_name}"


def job_chapters_dir(job_id: str) -> Path:
    path = job_root(job_id) / "chapters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_runtime_dir(job_id: str, tool_name: str) -> Path:
    safe_tool = tool_name.replace("..", "_").replace("/", "_").replace("\\", "_")
    path = job_root(job_id) / "runtime" / safe_tool
    path.mkdir(parents=True, exist_ok=True)
    return path
