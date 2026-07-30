"""
streamlit_app.py
================
User-facing demo for the AI4ALL Ignite chest X-ray fairness project.

IMPORTANT — HONESTY NOTE
------------------------
The team has NOT finished training a model yet. This app therefore runs in
PLACEHOLDER MODE: it produces clearly-labelled fake probabilities so the
interface, layout, and user flow can be built, demonstrated, and deployed now.

When the trained model is ready, the ONLY function that needs to change is
`predict()`. Set MODEL_READY = True and load your weights there. Nothing else
in the UI has to change.

Deploy on Streamlit Community Cloud:
  1. Put this file + requirements.txt in the repo root.
  2. streamlit.io -> "Create app" -> pick repo/branch, entrypoint streamlit_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Flip to True once a real trained model is wired into predict().
MODEL_READY = False

# The 14 NIH findings (No Finding is the absence of all of these).
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# Findings we explicitly flag as unreliable because of tiny training counts.
# (Hernia = 227 of 112,120 images = 0.2%.)
LOW_CONFIDENCE = {"Hernia": 227, "Pneumonia": 1431, "Fibrosis": 1686}

st.set_page_config(page_title="Chest X-ray Fairness Demo", page_icon="🫁", layout="wide")


# --------------------------------------------------------------------------
# Prediction  (THE ONLY PART THAT CHANGES WHEN THE REAL MODEL IS READY)
# --------------------------------------------------------------------------

def predict(image: Image.Image) -> dict:
    """Return {disease: probability}.

    Placeholder mode: deterministic pseudo-random numbers seeded by the image,
    so the same upload always gives the same output (feels real in a demo) but
    is NOT a real diagnosis.

    To go live:
        MODEL_READY = True
        model = load_your_densenet("model.pt")      # cache with st.cache_resource
        x = preprocess(image)                        # reuse preprocessing.py eval transform
        probs = torch.sigmoid(model(x))[0]
        return dict(zip(DISEASE_LABELS, probs.tolist()))
    """
    if MODEL_READY:
        raise NotImplementedError("Wire the trained model in here.")

    # --- placeholder only ---
    small = image.convert("L").resize((32, 32))
    seed = int(np.asarray(small).sum()) % (2**32)
    rng = np.random.default_rng(seed)
    raw = rng.random(len(DISEASE_LABELS)) ** 3  # skew low, like real prevalence
    return dict(zip(DISEASE_LABELS, raw.tolist()))


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("About this app")
    st.markdown(
        "A course demo (AI4ALL Ignite) that detects thoracic findings in chest "
        "X-rays **and** examines whether the model is equally reliable across "
        "patient groups."
    )
    st.markdown("**Model:** CNN / DenseNet-121 (transfer learning)")
    st.markdown("**Data:** NIH ChestX-ray14 — 112,120 images, 30,805 patients")
    st.divider()
    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="A finding is flagged as 'present' when its probability exceeds this value.",
    )
    st.caption("Team: Tianxin Dong · Junaid Pathan · Krisha Rathod · Junaid Pathan · Belema Roberts")


# --------------------------------------------------------------------------
# Header + honesty / safety banners
# --------------------------------------------------------------------------

st.title("🫁 Chest X-ray Disease Detection — with a Fairness Audit")

if not MODEL_READY:
    st.warning(
        "⚠️ **Placeholder mode.** The model is not trained yet, so the numbers "
        "below are **illustrative only** and are NOT real predictions. The "
        "interface is complete; real predictions appear once the model is wired in."
    )

st.error(
    "🚑 **Not a medical device.** This is a student project. It must never be "
    "used for real diagnosis or to make any health decision. Always consult a "
    "qualified clinician."
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
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

with col_results:
    st.subheader("2. Predicted findings")
    if not uploaded:
        st.info("Upload an image on the left to see predicted findings.")
    else:
        probs = predict(image)
        df = (
            pd.DataFrame({"Finding": list(probs.keys()),
                          "Probability": list(probs.values())})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )

        st.bar_chart(df.set_index("Finding")["Probability"], height=380)

        flagged = df[df["Probability"] >= threshold]
        if len(flagged) == 0:
            st.success(f"No finding exceeds the {threshold:.0%} threshold "
                       f"(closest to 'No Finding').")
        else:
            st.write(f"**Findings above the {threshold:.0%} threshold:**")
            for _, r in flagged.iterrows():
                note = ""
                if r["Finding"] in LOW_CONFIDENCE:
                    note = (f"  ⚠️ *low confidence — only "
                            f"{LOW_CONFIDENCE[r['Finding']]} training images*")
                st.write(f"- **{r['Finding']}** — {r['Probability']:.1%}{note}")


# --------------------------------------------------------------------------
# Fairness section — the heart of the project
# --------------------------------------------------------------------------

st.divider()
st.subheader("⚖️ Fairness & limitations")

fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.markdown("**Class imbalance**")
    st.markdown(
        "Findings range from Infiltration (19,894 images) to **Hernia (227)** — "
        "an ~88× gap. Rare findings are flagged as low-confidence above."
    )
with fc2:
    st.markdown("**Age skew**")
    st.markdown(
        "Disease prevalence rises with age (Hernia mean age 63 vs overall 47). "
        "We report recall per age band, not just overall."
    )
with fc3:
    st.markdown("**Label noise**")
    st.markdown(
        "Labels were NLP-mined from reports (~10% error). We use label smoothing "
        "and never treat outputs as ground truth."
    )

st.caption(
    "Fairness plan: once trained, this section will show measured recall for each "
    "sex and age group, so users can see where the model is least reliable — the "
    "core goal of the project."
)
