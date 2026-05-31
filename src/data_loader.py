from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Run `python -m src.generate_sample_data` first or place your dataset there."
        )
    return pd.read_csv(file_path)
