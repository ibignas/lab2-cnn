from pathlib import Path
from PIL import Image, UnidentifiedImageError, ImageOps
import random

random.seed(42)

target_size = (224, 244)
src_root = Path("./original_images")
dst_root = Path("./data")
test_ratio = 0.2
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

class_names = ["car", "bicycle", "motorcycle", "ship", "plane", "train"]

for split in ["train", "test"]:
    for cls in class_names:
        (dst_root / split / cls).mkdir(parents=True, exist_ok=True)

for cls_dir in sorted(src_root.iterdir()):
    if not cls_dir.is_dir():
        continue

    cls = cls_dir.name
    if cls not in class_names:
        continue

    files = [p for p in cls_dir.iterdir() if p.suffix.lower() in valid_exts]
    cleaned = []

    for p in files:
        try:
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img = ImageOps.pad(
                    img,
                    target_size,
                    method=Image.Resampling.BICUBIC,
                    color=(0, 0, 0),
                    centering=(0.5, 0.5),
                )
                cleaned.append((p, img.copy()))
        except (UnidentifiedImageError, OSError):
            continue
    random.shuffle(cleaned)
    n_test = int(len(cleaned) * test_ratio)

    for i, (src_path, img) in enumerate(cleaned):
        split = "test" if i < n_test else "train"
        out_path = dst_root / split / cls / f"{src_path.stem}.jpg"
        img.save(out_path, "JPEG", quality=95)