"""
streamlit_app.py
================
User-facing demo for the AI4ALL Ignite chest X-ray fairness project.

Runs the trained model (`checkpoints/simple_cnn_v2_full.pt`, test mean AUC 0.7647)
and — the point of the project — shows how reliable that prediction actually was
for patients of the selected sex and age group, using the measured recall from
`reports/fairness_recall_by_group.csv`.

The app deliberately reuses `preprocessing.get_transforms()` and `NIH_CLASSES`
rather than redefining them. Class order in particular must come from the training
code: the model outputs 14 logits in NIH_CLASSES order, and any other ordering
silently mislabels every prediction.

Deploy on Streamlit Community Cloud:
  1. Repo root must contain this file, requirements.txt, and the checkpoint.
  2. streamlit.io -> "Create app" -> pick repo/branch, entrypoint streamlit_app.py
"""

import pathlib

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from models import build_model
from preprocessing import NIH_CLASSES, get_transforms

ROOT = pathlib.Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints" / "simple_cnn_v2_full.pt"
RECALL_TABLE = ROOT / "reports" / "fairness_recall_by_group.csv"
THRESHOLD_TABLE = ROOT / "reports" / "fairness_thresholds.csv"

AGE_BANDS = ["<30", "30-49", "50-64", "65+"]

st.set_page_config(page_title="Chest X-ray Fairness Demo", page_icon="🫁", layout="wide")


# --------------------------------------------------------------------------
# Model + audit tables
# --------------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Rebuild the architecture from the hyperparameters stored in the checkpoint,
    so the app cannot drift out of sync with how the weights were trained."""
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    model = build_model(
        a.get("model", "simple_cnn"),
        dropout=a.get("dropout", 0.2),
        pretrained=False,
        num_blocks=a.get("num_blocks", 4),
        width=a.get("width", 2.0),
        pooling=a.get("pooling", "avgmax"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _, val_transform = get_transforms(a.get("img_size", 224))
    return model, val_transform, a


@st.cache_data
def load_audit():
    recall = pd.read_csv(RECALL_TABLE)
    thresholds = pd.read_csv(THRESHOLD_TABLE).set_index("finding")
    return recall, thresholds


@torch.no_grad()
def predict(image: Image.Image, model, transform) -> dict:
    x = transform(image.convert("RGB")).unsqueeze(0)
    probs = torch.sigmoid(model(x))[0]
    return dict(zip(NIH_CLASSES, probs.tolist()))


try:
    model, transform, ckpt_args = load_model()
    recall_df, thr_df = load_audit()
    MODEL_READY = True
except Exception as exc:                                    # noqa: BLE001
    MODEL_READY = False
    LOAD_ERROR = exc


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("About this app")
    st.markdown(
        "A course demo (AI4ALL Ignite) that detects thoracic findings in chest "
        "X-rays **and** reports whether the model is equally reliable across "
        "patient groups."
    )
    if MODEL_READY:
        n_params = sum(p.numel() for p in model.parameters())
        st.markdown(
            f"**Model:** SimpleCNN, {ckpt_args.get('num_blocks', 4)} conv blocks, "
            f"width {ckpt_args.get('width', 2.0)}×, {ckpt_args.get('pooling', 'avgmax')} "
            f"pooling — {n_params:,} parameters, trained from scratch"
        )
        st.markdown("**Test mean AUC:** 0.7647 on 17,131 held-out images")
    st.markdown("**Data:** NIH ChestX-ray14 — 112,120 images, 30,805 patients")

    st.divider()
    st.subheader("Patient context")
    st.caption(
        "Used only to look up how well the model performed on this group in our "
        "audit. It is not fed to the model."
    )
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    age_band = st.selectbox("Age band", AGE_BANDS, index=2)

    st.divider()
    st.caption("Team: Tianxin Dong · Krisha Rathod · Junaid Pathan · Belema Roberts")


# --------------------------------------------------------------------------
# Header + safety banners
# --------------------------------------------------------------------------

st.title("🫁 Chest X-ray Disease Detection — with a Fairness Audit")

if not MODEL_READY:
    st.error(
        f"Model failed to load: `{LOAD_ERROR}`. The app will not fabricate "
        "predictions in place of a working model."
    )
    st.stop()

st.error(
    "🚑 **Not a medical device.** This is a student project trained from scratch on "
    "a research dataset with NLP-derived labels. Precision is low on most findings. "
    "It must never be used for diagnosis or any health decision."
)


# --------------------------------------------------------------------------
# Main interaction
# --------------------------------------------------------------------------

col_upload, col_results = st.columns([1, 1.3])

with col_upload:
    st.subheader("1. Upload a chest X-ray")
    uploaded = st.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

with col_results:
    st.subheader("2. Predicted findings")
    if not uploaded:
        st.info("Upload an image on the left to see predicted findings.")
    else:
        probs = predict(image, model, transform)
        df = pd.DataFrame({"Finding": list(probs), "Probability": list(probs.values())})
        df["Threshold"] = df["Finding"].map(thr_df["threshold"])
        df["Flagged"] = df["Probability"] >= df["Threshold"]
        df = df.sort_values("Probability", ascending=False).reset_index(drop=True)

        st.bar_chart(df.set_index("Finding")["Probability"], height=360)
        st.caption(
            "Each finding uses its own decision threshold, fitted on the validation "
            "split to reach 80% recall. A single shared cut-off would drive recall to "
            "zero on the rarer findings."
        )

        flagged = df[df["Flagged"]]
        if flagged.empty:
            st.success("No finding exceeds its decision threshold.")
        else:
            st.write("**Findings above their threshold:**")
            for _, r in flagged.iterrows():
                st.write(f"- **{r['Finding']}** — {r['Probability']:.1%} "
                         f"(threshold {r['Threshold']:.2f})")


# --------------------------------------------------------------------------
# Fairness — measured reliability for this patient group
# --------------------------------------------------------------------------

st.divider()
st.subheader("⚖️ How reliable is this, for this patient?")

group = f"{sex} {age_band}"
st.markdown(
    f"Measured on our held-out test set for **{group}** patients. "
    "Recall answers: of the patients in this group who genuinely had the finding, "
    "what fraction did the model flag? The interval is a 95% Wilson interval; "
    "*n* is the number of positive cases it rests on."
)

g = recall_df[recall_df["group"] == group].copy()
overall = (recall_df.groupby("finding")
           .apply(lambda x: (x["recall"] * x["n_positive"]).sum() / x["n_positive"].sum(),
                  include_groups=False)
           .rename("all_groups"))
g = g.merge(overall, on="finding")
g["vs_overall"] = g["recall"] - g["all_groups"]
g = g.sort_values("recall")

show = pd.DataFrame({
    "Finding": g["finding"],
    "Recall for this group": g["recall"].map("{:.0%}".format),
    "95% CI": [f"{lo:.0%} – {hi:.0%}" for lo, hi in zip(g["ci_low"], g["ci_high"])],
    "n": g["n_positive"],
    "vs. all patients": g["vs_overall"].map("{:+.0%}".format),
    "Reliable?": ["yes" if t == 1 else "no — see note" for t in g["tier"]],
})
st.dataframe(show, hide_index=True, use_container_width=True)

worst = g.iloc[0]
if worst["tier"] == 1:
    st.warning(
        f"Weakest for this group: **{worst['finding']}** — the model caught "
        f"{worst['recall']:.0%} of {group} patients who had it, against "
        f"{worst['all_groups']:.0%} across all patients."
    )

st.caption(
    "Findings marked *no — see note* reach their recall largely by flagging a very "
    "large share of images, so differences between groups there are not meaningful. "
    "Five findings are excluded entirely because some group had too few positive "
    "cases to estimate recall — Hernia has 10 in the whole test set."
)


# --------------------------------------------------------------------------
# Known limitations
# --------------------------------------------------------------------------

st.divider()
st.subheader("Known limitations")

lc1, lc2, lc3 = st.columns(3)
with lc1:
    st.markdown("**Class imbalance**")
    st.markdown(
        "Findings range from Infiltration (19,894 images) to **Hernia (227)** — an "
        "88× gap. Rare findings are excluded from the fairness table above rather "
        "than reported with unusable uncertainty."
    )
with lc2:
    st.markdown("**Measured recall gaps**")
    st.markdown(
        "Pneumothorax is caught in 89% of women aged 30-49 but 64% of men in the "
        "same band. Gaps are finding-specific — no group is uniformly underserved."
    )
with lc3:
    st.markdown("**Label noise & confounds**")
    st.markdown(
        "Labels were NLP-mined from reports (~10% error). AP films are taken bedside "
        "on sicker patients, so view position is a confound we flagged but did not "
        "remove."
    )

st.caption(
    "Full analysis: `fairness_eval.ipynb` · Written summary: "
    "`reports/fairness_audit_report.md`"
)
