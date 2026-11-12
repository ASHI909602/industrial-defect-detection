# 🏭 Industrial Defect Detection (Deep Learning)

Product image upload karo → preprocess → CNN (MobileNetV2) → **Defect / No Defect**,
**defect category**, **confidence** aur **evaluation report**.

## Pipeline
1. Image input (Streamlit upload)
2. Preprocess: RGB, autocontrast, resize 224x224, normalize (+ train augmentation)
3. CNN: MobileNetV2 transfer learning
4. Output: status, category, confidence, class probabilities
5. Report: accuracy, precision/recall/F1, confusion matrix

## Dataset
Recommended: [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm)
(6 categories: crazing, inclusions, patches, pitted_surface, rolled-in_scale, scratches)
ya [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad).

`raw_data/` me aise rakho (ek folder = ek class, **no_defect** zaroor ho):
```
raw_data/
  no_defect/  crazing/  inclusions/  patches/  pitted_surface/  rolled-in_scale/  scratches/
```
NEU me good samples nahi hote; no_defect ke liye MVTec ke "good" images ya apni
factory ke OK product images use karo.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_data     # train/val/test split
python -m src.train            # models/best_model.pt
python -m src.evaluate         # reports/metrics.json + confusion_matrix.png
streamlit run app.py
```

## Deploy
- **Streamlit Community Cloud**: repo GitHub par push → share.streamlit.io → New app → `app.py`.
- **Docker / Hugging Face Spaces / Render**:
  `docker build -t defect-app . && docker run -p 8501:8501 defect-app`
- Model + `reports/` commit karna na bhoolo, warna app me model nahi milega.
