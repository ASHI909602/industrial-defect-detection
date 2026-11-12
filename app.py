import json

import pandas as pd  # streamlit ke saath aata hai
import streamlit as st

from src.config import MODEL_PATH, REPORT_DIR
from src.predict import load_model, predict

st.set_page_config(page_title="Industrial Defect Detection", page_icon="🏭", layout="wide")
st.title("🏭 Industrial Defect Detection")


@st.cache_resource
def get_model():
    return load_model()


if not MODEL_PATH.exists():
    st.error("Model nahi mila. Pehle `python -m src.train` chalao.")
    st.stop()

model, classes = get_model()
tab1, tab2 = st.tabs(["🔍 Inspect Product", "📊 Evaluation Report"])

with tab1:
    file = st.file_uploader("Product image upload karo", type=["jpg", "jpeg", "png", "bmp"])
    if file:
        left, right = st.columns(2)
        left.image(file, caption="Input image", use_container_width=True)
        res = predict(model, classes, file)
        with right:
            if res["status"] == "DEFECT":
                st.error(f"❌ DEFECT — {res['category']}")
            else:
                st.success("✅ NO DEFECT")
            st.metric("Confidence", f"{res['confidence'] * 100:.1f}%")
            st.progress(float(res["confidence"]))
            st.caption("Class probabilities")
            st.bar_chart(pd.Series(res["probabilities"]))

with tab2:
    mfile = REPORT_DIR / "metrics.json"
    if mfile.exists():
        m = json.loads(mfile.read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{m['accuracy'] * 100:.1f}%")
        c2.metric("Macro F1", f"{m['macro_f1']:.3f}")
        c3.metric("Defect Recall", f"{m['binary_defect']['recall']:.3f}")
        st.subheader("Per-class metrics")
        df = pd.DataFrame(m["per_class"]).T[["precision", "recall", "f1-score", "support"]]
        st.dataframe(df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                      "f1-score": "{:.3f}", "support": "{:.0f}"}))
        cm = REPORT_DIR / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), caption="Confusion Matrix", width=600)
    else:
        st.info("Report nahi mili. `python -m src.evaluate` chalao.")
