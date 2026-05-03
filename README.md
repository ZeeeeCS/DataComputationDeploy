# HAR Activity Classifier — Streamlit App

A Streamlit web app for the Human Activity Recognition (HAR) ML pipeline
built with custom sklearn transformers + SVM + Optuna hyperparameter tuning.

---

## 📁 Project Structure

```
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## ☁️ Deploy on Streamlit Community Cloud (free)

1. **Push to GitHub**

   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Initial commit"
   # Create a new repo on github.com, then:
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app** → select your repo → set **Main file path** to `app.py` → **Deploy**.

4. Your app will be live at `https://<your-app>.streamlit.app` in ~2 minutes.

---

## ☁️ Deploy on Hugging Face Spaces (free)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the SDK.
3. Upload `app.py` and `requirements.txt`.
4. The Space builds and serves your app automatically.

---

## 📊 App Features

| Feature | Details |
|---|---|
| File upload | Train + Test CSV via sidebar |
| EDA | Class distribution, correlation matrix, missing/duplicate stats |
| Preprocessing | Configurable variance / correlation / PCA thresholds |
| Hyperparameter tuning | Manual sliders or Optuna (configurable trials) |
| Results | CV accuracy, test accuracy, classification report, confusion matrix |
| Dimensionality report | Feature count at each pipeline stage |

---

## 📝 Expected CSV Format

Both CSVs must contain:
- An **`Activity`** column (target label)
- Optionally a **`subject`** column (dropped automatically)
- All remaining columns treated as features
"# DataComputationDeploy" 
"# DataComputationDeploy" 
