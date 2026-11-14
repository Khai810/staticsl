from pathlib import Path


def analyze_validation(root: Path, run: Path, cfg: dict) -> dict:
    """Khóa model, sweep validation rồi xuất metrics, figures và gallery."""
    # TODO: verify checkpoint -> predictions -> sweep -> inference.json -> report.
    raise NotImplementedError


def regenerate_figures(run: Path) -> None:
    """Vẽ lại từ artifact thật đã lưu, không inference hay chấm test."""
    # TODO: CSV/JSON -> PNG300/SVG/captions.
    raise NotImplementedError
