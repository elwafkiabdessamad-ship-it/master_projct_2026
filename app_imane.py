import streamlit as st
from app_utils import (
    load_pipeline, validate_smiles, predict_toxicity, get_model_metrics,
    get_feature_importance, mol_to_image, EXAMPLE_MOLS,
)

st.set_page_config(page_title="Imane — QSAR Toxicity Predictor", layout="wide")
st.title("🧪 Imane's QSAR Toxicity Predictor")
st.caption("Binary classification model for acute aquatic toxicity (log_LC50 < 1.0 → toxic)")

pipeline = load_pipeline("classical_models_Imane.pkl")
data_info = pipeline["data_info"]
top3 = pipeline["top3"]

with st.sidebar:
    st.header("About the dataset")
    st.metric("Molecules", data_info["n_molecules"], help="Total curated molecules")
    st.metric(
        "Toxic / Non-toxic",
        f'{data_info["n_toxic"]} / {data_info["n_non_toxic"]}',
        help=f'{data_info["n_toxic"] / data_info["n_molecules"] * 100:.1f}% toxic',
    )
    st.metric(
        "Features (after filter)",
        data_info["n_features_selected"],
        help=f'{data_info["n_features_raw"]} raw → {data_info["n_features_selected"]} after VarianceThreshold',
    )

    st.divider()
    st.subheader("Top classifiers (AUC-ROC)")
    for rank, r in enumerate(sorted(pipeline["results"], key=lambda x: x["auc_roc"], reverse=True)[:5], 1):
        st.markdown(f"**{rank}.** {r['model']} — `{r['auc_roc']:.4f}`")

    st.divider()
    st.markdown(
        """
    **How it works**
    1. Enter a SMILES string
    2. Morgan fingerprint (2048 bits) + 217 RDKit descriptors
    3. VarianceThreshold filter (same as training)
    4. Classifier predicts toxicity
    """
    )

tab_predict, tab_compare = st.tabs(["🔬 Predict", "📊 Model comparison"])

with tab_predict:
    col_input, col_viz = st.columns([3, 2])

    with col_input:
        st.subheader("Enter a molecule")

        example = st.selectbox(
            "Try an example", ["Custom"] + list(EXAMPLE_MOLS.keys()), index=0
        )
        default_smiles = EXAMPLE_MOLS.get(example, "")

        smiles = st.text_input(
            "SMILES",
            value=default_smiles,
            placeholder="e.g. CCO, c1ccccc1, O=C(O)c1ccccc1",
            label_visibility="collapsed",
        )
        model_name = st.selectbox("Classifier", top3, index=0)

    with col_viz:
        if smiles.strip():
            img = mol_to_image(smiles.strip(), size=(350, 250))
            if img:
                st.image(img, caption="Molecule structure", width=350)

    if st.button("Predict toxicity", type="primary", use_container_width=True) and smiles.strip():
        canonical = validate_smiles(smiles.strip())
        if canonical is None:
            st.error("Invalid SMILES string. Please check and try again.")
            st.stop()

        result = predict_toxicity(canonical, pipeline, model_name)
        if result is None:
            st.error("Error computing features.")
            st.stop()

        pred, proba = result

        st.divider()

        cols = st.columns([1, 1, 1, 1])
        cols[0].metric("Prediction", "Toxic" if pred == 1 else "Non-toxic", delta=None)
        cols[1].metric("Confidence", f"{proba:.1%}")
        cols[2].metric("Model", model_name)
        cols[3].metric("p(toxic)", f"{proba:.4f}")

        st.progress(proba)

        if pred == 1:
            st.warning(f"⚠️ Predicted **toxic** — p(toxic) = {proba:.3f}, log_LC50 likely < 1.0")
        else:
            st.success(f"✅ Predicted **non-toxic** — p(toxic) = {proba:.3f}, log_LC50 likely ≥ 1.0")

        with st.expander("Canonical SMILES & predicted log_LC50"):
            st.code(canonical)
            predicted_lc50 = 1.0 - (proba - 0.5) * 4
            st.markdown(f"Approximate log_LC50: **{predicted_lc50:.2f}**")

with tab_compare:
    st.subheader("Model performance on held-out test set")

    metrics_df = []
    for r in pipeline["results"]:
        metrics_df.append(
            {
                "Model": r["model"],
                "AUC-ROC": f"{r['auc_roc']:.4f}",
                "Accuracy": f"{r['accuracy']:.4f}",
                "F1 Score": f"{r['f1']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "Specificity": f"{r['specificity']:.4f}",
            }
        )
    import pandas as pd

    st.dataframe(
        pd.DataFrame(metrics_df).sort_values("AUC-ROC", ascending=False),
        hide_index=True,
        width="stretch",
    )

    st.divider()
    st.subheader("Feature importance (top 15)")

    fi_model = st.selectbox("Model", [m for m in top3 if hasattr(pipeline["models"][m], "feature_importances_")])
    fi = get_feature_importance(pipeline, fi_model, top_n=15)
    if fi:
        names, vals = zip(*fi)
        import streamlit as st_chart
        st.bar_chart(
            pd.DataFrame({"importance": vals}, index=names),
            width="stretch",
        )
