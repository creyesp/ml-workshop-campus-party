"""Compute and print sha256 digests for the pinned datasets.

Run: ``.venv/bin/python experiments/00_pin_data.py``
Paste the printed digests into ``floodit.config.DATA_HASHES``.
"""
from floodit.data.load import dataset_path, sha256_of

if __name__ == "__main__":
    for name in ("train", "test", "raw"):
        print(f'    "{name}": "{sha256_of(dataset_path(name))}",')
