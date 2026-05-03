import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import streamlit as st
import io

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAR Activity Classifier",
    page_icon="🏃",
    layout="wide",
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Custom Transformers ───────────────────────────────────────────────────────
class DuplicateFeatureFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.keep_cols_ = X.T.drop_duplicates().T.columns
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X[self.keep_cols_]


class VarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.selector_ = VarianceThreshold(self.threshold)
        self.selector_.fit(X)
        self.keep_cols_ = X.columns[self.selector_.get_support()]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X[self.keep_cols_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        y = pd.Series(y)
        corr_matrix = X.corr().abs()
        feature_corr = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        mi_scores = mutual_info_classif(X, y)
        target_corr = pd.Series(mi_scores, index=X.columns)
        to_drop = set()
        for f1 in feature_corr.columns:
            for f2 in feature_corr.index:
                if feature_corr.loc[f2, f1] > self.threshold:
                    if target_corr[f1] < target_corr[f2]:
                        to_drop.add(f1)
                    else:
                        to_drop.add(f2)
        self.keep_cols_ = [c for c in X.columns if c not in to_drop]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X[self.keep_cols_]


class GroupwisePCA(BaseEstimator, TransformerMixin):
    def __init__(self, freq_var=0.90, time_var=0.95, random_state=RANDOM_SEED):
        self.freq_var = freq_var
        self.time_var = time_var
        self.random_state = random_state

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = X.columns
        self.groupwise_report_ = []
        group_map = (
            X.columns.to_series()
            .groupby(X.columns.map(lambda c: c.split("-")[0].split("(")[0]))
            .apply(list)
        )
        self.group_models_ = []
        for group_name, cols in group_map.items():
            Xg = X[cols]
            scaler = StandardScaler()
            Xg_scaled = scaler.fit_transform(Xg)
            var = self.freq_var if group_name.startswith("f") else self.time_var
            pca = PCA(n_components=var, random_state=self.random_state)
            pca.fit(Xg_scaled)
            self.groupwise_report_.append({
                "group": group_name,
                "n_features_before": Xg.shape[1],
                "n_features_after": pca.n_components_,
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
            })
            self.group_models_.append({"cols": cols, "scaler": scaler, "pca": pca})
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        parts = []
        for item in self.group_models_:
            Xg = X[item["cols"]]
            Xg_scaled = item["scaler"].transform(Xg)
            parts.append(item["pca"].transform(Xg_scaled))
        return np.hstack(parts)

    def get_groupwise_report_df(self):
        return pd.DataFrame(self.groupwise_report_)


def build_model(params):
    prep = Pipeline([
        ("dup",       DuplicateFeatureFilter()),
        ("var",       VarianceFilter(params["var_th"])),
        ("corr",      CorrelationFilter(params["corr_th"])),
        ("group_pca", GroupwisePCA(params["group_pca"], params["group_pca"])),
        ("final_pca", PCA(params["final_pca"], random_state=RANDOM_SEED)),
    ])
    model = Pipeline([
        ("prep", prep),
        ("svm", SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            random_state=RANDOM_SEED,
        )),
    ])
    return model


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏃 Human Activity Recognition — ML Pipeline")
st.markdown("Upload your train/test CSVs, configure parameters, and train the SVM pipeline.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📂 Data Upload")
    train_file = st.file_uploader("Train CSV", type="csv")
    test_file  = st.file_uploader("Test CSV",  type="csv")

    st.subheader("🔬 Preprocessing Thresholds")
    var_th   = st.slider("Variance threshold",    0.0,  0.05, 0.01,  0.001, format="%.3f")
    corr_th  = st.slider("Correlation threshold", 0.80, 1.00, 0.95,  0.01,  format="%.2f")
    gpca     = st.slider("Group PCA variance",    0.80, 0.99, 0.90,  0.01,  format="%.2f")
    fpca     = st.slider("Final PCA variance",    0.90, 0.99, 0.98,  0.01,  format="%.2f")

    st.subheader("🤖 SVM Hyperparameters")
    C_val    = st.number_input("C (regularisation)", 0.01, 200.0, 1.0,  step=0.1)
    kernel   = st.selectbox("Kernel", ["rbf", "linear", "poly"])
    gamma    = st.selectbox("Gamma",  ["scale", "auto", "0.1", "0.01", "0.001"])
    gamma    = float(gamma) if gamma not in ("scale", "auto") else gamma

    st.subheader("🔍 Optuna Tuning")
    use_optuna  = st.checkbox("Enable Optuna tuning", value=False)
    n_trials    = st.number_input("Number of trials", 1, 50, 5, disabled=not use_optuna)
    n_cv_splits = st.slider("CV folds", 2, 10, 3)

    run_btn = st.button("🚀 Train Model", use_container_width=True, type="primary")

# ── Main area placeholders ────────────────────────────────────────────────────
if not run_btn:
    st.info("👈  Upload your data and press **Train Model** to begin.")
    st.stop()

if train_file is None or test_file is None:
    st.error("Please upload both **Train** and **Test** CSV files.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

# Drop 'subject' column if present
for col in ["subject"]:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

X_train = train_df.drop(columns=["Activity"])
y_train = train_df["Activity"]
X_test  = test_df.drop(columns=["Activity"])
y_test  = test_df["Activity"]

# ── EDA ───────────────────────────────────────────────────────────────────────
st.header("📊 Exploratory Data Analysis")
col1, col2, col3 = st.columns(3)
col1.metric("Train samples",  X_train.shape[0])
col2.metric("Test samples",   X_test.shape[0])
col3.metric("Features",       X_train.shape[1])

tab_eda1, tab_eda2, tab_eda3 = st.tabs(["Class Distribution", "Correlation Matrix", "Missing / Duplicates"])

with tab_eda1:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = y_train.value_counts().index
    sns.countplot(y=y_train, order=order, palette="Set2", ax=ax)
    ax.set_title("Activity Distribution (Train)")
    ax.set_xlabel("Count")
    st.pyplot(fig)
    plt.close(fig)

with tab_eda2:
    top20 = X_train.var().nlargest(25).index.tolist()
    corr  = X_train[top20].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.01, ax=ax)
    ax.set_title("Correlation Matrix — Top 25 High-Variance Features")
    st.pyplot(fig)
    plt.close(fig)

with tab_eda3:
    missing_train = train_df.isnull().sum().sum()
    dup_rows      = train_df.duplicated().sum()
    nzv           = (X_train.var() < 0.01).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Missing values",          missing_train)
    c2.metric("Duplicate rows",          dup_rows)
    c3.metric("Near-zero variance cols", nzv)

# ── Training ──────────────────────────────────────────────────────────────────
st.header("🏋️ Model Training")
cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=RANDOM_SEED)

base_params = dict(
    var_th=var_th, corr_th=corr_th,
    group_pca=gpca, final_pca=fpca,
    C=C_val, kernel=kernel, gamma=gamma,
)

if use_optuna:
    st.info(f"Running Optuna with {n_trials} trials — this may take a while…")
    prog = st.progress(0)
    trial_counter = [0]

    def objective(trial):
        p = dict(
            var_th    = trial.suggest_float("var_th",    0.0,  0.02),
            corr_th   = trial.suggest_float("corr_th",   0.90, 0.97),
            group_pca = trial.suggest_float("group_pca", 0.90, 0.99),
            final_pca = trial.suggest_float("final_pca", 0.95, 0.99),
            C         = trial.suggest_float("C",         0.1,  100, log=True),
            kernel    = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            gamma     = trial.suggest_categorical("gamma",  ["scale", 0.1, 0.01, 0.001]),
        )
        score = cross_val_score(build_model(p), X_train, y_train,
                                cv=cv, scoring="accuracy", n_jobs=-1).mean()
        trial_counter[0] += 1
        prog.progress(trial_counter[0] / n_trials)
        return score

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=int(n_trials))
    best_params = study.best_params
    st.success(f"✅ Optuna best CV accuracy: **{study.best_value:.4f}**")

    trials_df = study.trials_dataframe()[["number", "value"]].rename(
        columns={"number": "Trial", "value": "CV Accuracy"}
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(trials_df["Trial"], trials_df["CV Accuracy"], marker="o")
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Optuna Optimization History")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Best Parameters Found")
    st.json(best_params)
else:
    best_params = base_params

# ── Fit final model ────────────────────────────────────────────────────────────
with st.spinner("Fitting final model…"):
    best_model = build_model(best_params)
    cv_scores  = cross_val_score(best_model, X_train, y_train,
                                 cv=cv, scoring="accuracy", n_jobs=-1)
    best_model.fit(X_train, y_train)

mean_cv = cv_scores.mean()
std_cv  = cv_scores.std()

# ── Dimensionality reduction report ───────────────────────────────────────────
st.header("🔬 Dimensionality Reduction Report")
prep  = best_model.named_steps["prep"]
X_tmp = X_train.copy()

X_tmp      = prep.named_steps["dup"].transform(X_tmp);       after_dup   = X_tmp.shape[1]
X_tmp      = prep.named_steps["var"].transform(X_tmp);       after_var   = X_tmp.shape[1]
X_tmp      = prep.named_steps["corr"].transform(X_tmp);      after_corr  = X_tmp.shape[1]
X_tmp      = prep.named_steps["group_pca"].transform(X_tmp); after_group = X_tmp.shape[1]
X_tmp      = prep.named_steps["final_pca"].transform(X_tmp); after_final = X_tmp.shape[1]

reduction_df = pd.DataFrame({
    "Stage":    ["Original", "Duplicate filter", "Variance filter",
                 "Correlation filter", "Group-wise PCA", "Final PCA"],
    "Features": [X_train.shape[1], after_dup, after_var,
                 after_corr, after_group, after_final],
})

col_a, col_b = st.columns([1, 2])
with col_a:
    st.dataframe(reduction_df, use_container_width=True, hide_index=True)
with col_b:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(reduction_df["Stage"], reduction_df["Features"], color=sns.color_palette("Blues_d", len(reduction_df)))
    ax.set_xticklabels(reduction_df["Stage"], rotation=30, ha="right")
    ax.set_ylabel("# Features")
    ax.set_title("Feature Count Through Pipeline")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Group-wise PCA Detail")
st.dataframe(
    prep.named_steps["group_pca"].get_groupwise_report_df()
        .style.format({"explained_variance": "{:.3f}"}),
    use_container_width=True, hide_index=True,
)

# ── Results ───────────────────────────────────────────────────────────────────
st.header("📈 Evaluation Results")

y_pred_train = best_model.predict(X_train)
y_pred_test  = best_model.predict(X_test)
test_acc     = accuracy_score(y_test, y_pred_test)

m1, m2, m3 = st.columns(3)
m1.metric("CV Accuracy (Train)", f"{mean_cv:.4f}", f"±{std_cv:.4f}")
m2.metric("Accuracy on Test Set", f"{test_acc:.4f}")
m3.metric("Feature reduction", f"{X_train.shape[1]} → {after_final}")

tab_train, tab_test = st.tabs(["Train Evaluation", "Test Evaluation"])

def show_results(y_true, y_pred, classes, cmap, title_suffix):
    report_str = classification_report(y_true, y_pred)
    st.text(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap=cmap,
                linewidths=0.5, linecolor="gray", square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f"Confusion Matrix — {title_suffix}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab_train:
    show_results(y_train, y_pred_train, best_model.classes_, "Blues", "Train Set")

with tab_test:
    show_results(y_test, y_pred_test, best_model.classes_, "Oranges", "Test Set")

# ── Summary ───────────────────────────────────────────────────────────────────
st.header("🏁 Final Conclusion")
st.success(
    f"**CV Accuracy (Train):** {mean_cv:.4f} ± {std_cv:.4f}  \n"
    f"**Test Accuracy:** {test_acc:.4f}"
)
