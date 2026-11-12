"""Inference helper: image -> defect/no-defect + category + confidence."""
import torch

from .config import MODEL_PATH, NO_DEFECT_CLASS
from .model import build_model
from .preprocess import eval_transform, load_image


def load_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu")
    classes = ckpt["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


@torch.no_grad()
def predict(model, classes, image_src) -> dict:
    img = load_image(image_src)
    x = eval_transform(img).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1)[0].tolist()
    prob_map = dict(zip(classes, probs))

    p_ok = prob_map.get(NO_DEFECT_CLASS, 0.0)
    is_defect = p_ok < 0.5

    if is_defect:
        defect_probs = {c: p for c, p in prob_map.items() if c != NO_DEFECT_CLASS}
        category = max(defect_probs, key=defect_probs.get)
        confidence = 1 - p_ok
    else:
        category = None
        confidence = p_ok

    return {
        "status": "DEFECT" if is_defect else "NO DEFECT",
        "category": category,
        "confidence": confidence,
        "probabilities": prob_map,
    }


if __name__ == "__main__":
    import sys
    m, c = load_model()
    print(predict(m, c, sys.argv[1]))
