"""Hash-pinned dataset loading.

The CSVs are a frozen workshop snapshot. Loading verifies the file's sha256
against the digest pinned in :data:`floodit.config.DATA_HASHES` so an
experiment cannot silently train on altered data.
"""
import hashlib
from pathlib import Path

import pandas as pd

from floodit import config

_FILES = {
    "train": "users_train.csv",
    "test": "users_test.csv",
    "raw": "users_raw.csv",
}


def sha256_of(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks (constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(name: str) -> Path:
    if name not in _FILES:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(_FILES)}")
    return config.DATA_DIR / _FILES[name]


def load_dataset(name: str, verify: bool = True) -> pd.DataFrame:
    """Load a pinned dataset CSV as a DataFrame.

    Parameters
    ----------
    name : {"train", "test", "raw"}
    verify : if True, assert the file sha256 matches ``config.DATA_HASHES``.
        Raises KeyError if the hash has not been pinned yet (Task 2).
    """
    path = dataset_path(name)
    if verify:
        expected = config.DATA_HASHES[name]
        actual = sha256_of(path)
        if actual != expected:
            raise ValueError(
                f"Hash mismatch for {name}: expected {expected}, got {actual}"
            )
    df = pd.read_csv(path)
    # Timestamps are tz-aware (+00:00); parse to datetime64[ns, UTC] explicitly.
    df["user_first_engagement"] = pd.to_datetime(
        df["user_first_engagement"], utc=True, format="ISO8601"
    )
    return df
