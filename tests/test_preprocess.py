from PIL import Image

from src.config import IMG_SIZE
from src.preprocess import eval_transform


def test_output_shape():
    img = Image.new("L", (300, 200), 128).convert("RGB")
    assert eval_transform(img).shape == (3, IMG_SIZE, IMG_SIZE)
