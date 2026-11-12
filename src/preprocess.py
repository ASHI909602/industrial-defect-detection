"""Image preprocessing: train aur inference dono me same pipeline."""
from PIL import Image, ImageOps
from torchvision import transforms as T

from .config import IMG_SIZE, MEAN, STD


def load_image(src) -> Image.Image:
    """Path ya file-like object se RGB image load karo."""
    return Image.open(src).convert("RGB")


def enhance(img: Image.Image) -> Image.Image:
    """Contrast normalize karta hai taaki scratches/cracks zyada visible ho."""
    return ImageOps.autocontrast(img, cutoff=1)


eval_transform = T.Compose([
    T.Lambda(enhance),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

train_transform = T.Compose([
    T.Lambda(enhance),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
