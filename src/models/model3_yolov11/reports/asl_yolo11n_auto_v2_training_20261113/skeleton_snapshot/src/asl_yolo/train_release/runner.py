from pathlib import Path


def run_training(root: Path, cfg: dict, mode: str, resume_run: Path | None = None, checkpoint: Path | None = None, stop_after: int | None = None) -> Path:
    """Chạy mới hoặc resume rõ run, chỉ khi mọi cổng đã đạt."""
    # TODO: preflight -> run lock -> train -> checkpoint verification -> status.
    raise NotImplementedError


def resume_probe(root: Path, cfg: dict) -> dict:
    """Dừng sau epoch đã lưu và resume trong tiến trình GPU mới."""
    # TODO: process1 stop1/total4 -> process2 resume -> verify optimizer/history/model.
    raise NotImplementedError
