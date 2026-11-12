"""raw_data/<class>/*.jpg  ->  data/{train,val,test}/<class>/*.jpg (70/15/15)

Usage: python -m src.prepare_data
"""
import random
import shutil

from .config import DATA_DIR, RAW_DIR, SEED

EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    random.seed(SEED)
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    for cls_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in EXT]
        random.shuffle(files)
        n = len(files)
        splits = {
            "train": files[: int(0.7 * n)],
            "val": files[int(0.7 * n): int(0.85 * n)],
            "test": files[int(0.85 * n):],
        }
        for split, items in splits.items():
            out = DATA_DIR / split / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy(f, out / f.name)
        print(f"{cls_dir.name}: {n} images")


if __name__ == "__main__":
    main()
