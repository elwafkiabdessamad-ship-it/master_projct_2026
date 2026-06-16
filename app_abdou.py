import gradio as gr
import pandas as pd
from io import BytesIO
from app_utils import (
    load_pipeline, validate_smiles, predict_toxicity, get_model_metrics,
    get_feature_importance, mol_to_image, EXAMPLE_MOLS,
)

pipeline = load_pipeline("classical_models_Abdou.pkl")
data_info = pipeline["data_info"]
top3 = pipeline["top3"]

MODEL_KWARGS = {}
for r in pipeline["results"]:
    MODEL_KWARGS[r["model"]] = {
        "AUC-ROC": f"{r['auc_roc']:.4f}",
        "F1": f"{r['f1']:.4f}",
        "Accuracy": f"{r['accuracy']:.4f}",
    }


def predict(smiles: str, model_name: str):
    if not smiles or not smiles.strip():
        return None, "Please enter a SMILES string.", "", "", 0.0, "", None

    canonical = validate_smiles(smiles.strip())
    if canonical is None:
        return None, "Invalid SMILES string.", "", "", 0.0, "", None

    result = predict_toxicity(canonical, pipeline, model_name)
    if result is None:
        return None, "Error computing features.", "", "", 0.0, "", None

    pred, proba = result

    label = "Toxic" if pred == 1 else "Non-toxic"
    confidence = f"{proba:.1%}"
    score = proba

    msg = (
        f"⚠️ Predicted **toxic** — p(toxic) = {proba:.3f}, log_LC50 likely < 1.0"
        if pred == 1
        else f"✅ Predicted **non-toxic** — p(toxic) = {proba:.3f}, log_LC50 likely ≥ 1.0"
    )

    kw = MODEL_KWARGS.get(model_name, {})
    details = (
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Model | {model_name} |\n"
        f"| AUC-ROC | {kw.get('AUC-ROC', '-')} |\n"
        f"| F1 Score | {kw.get('F1', '-')} |\n"
        f"| Canonical SMILES | `{canonical}` |\n"
    )

    img = mol_to_image(canonical, size=(350, 250))
    return img, label, confidence, msg, score, f"p(toxic) = {proba:.4f}", details


def update_example(example_name: str):
    return EXAMPLE_MOLS.get(example_name, "")


with gr.Blocks(title="Abdou — QSAR Toxicity Predictor") as demo:
    gr.Markdown("# 🧪 Abdou's QSAR Toxicity Predictor")
    gr.Markdown("Binary classification model for acute aquatic toxicity (log_LC50 < 1.0 → toxic)")

    with gr.Row():
        with gr.Column(scale=2):
            smiles_input = gr.Textbox(
                label="SMILES",
                placeholder="e.g. CCO, c1ccccc1, O=C(O)c1ccccc1",
            )
            example_selector = gr.Dropdown(
                choices=["Custom"] + list(EXAMPLE_MOLS.keys()),
                value="Custom",
                label="Try an example",
            )
            model_dropdown = gr.Dropdown(
                choices=top3,
                value="RandomForest",
                label="Classifier",
            )
            predict_btn = gr.Button("Predict toxicity", variant="primary", size="lg")

        with gr.Column(scale=1):
            mol_image = gr.Image(label="Molecule structure", height=260)

    example_selector.change(fn=update_example, inputs=example_selector, outputs=smiles_input)

    with gr.Accordion("Prediction result", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                prediction = gr.Textbox(label="Prediction")
                confidence = gr.Textbox(label="Confidence")
                proba_display = gr.Textbox(label="Probability")
            with gr.Column(scale=2):
                detail_msg = gr.Markdown()

    with gr.Row():
        prob_score = gr.Number(value=0.0, label="Probability score", minimum=0.0, maximum=1.0)
        details_table = gr.Markdown()

    predict_btn.click(
        fn=predict,
        inputs=[smiles_input, model_dropdown],
        outputs=[mol_image, prediction, confidence, detail_msg, prob_score, proba_display, details_table],
    )

    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                f"**Dataset:** {data_info['n_molecules']} molecules "
                f"({data_info['n_toxic']} toxic, {data_info['n_non_toxic']} non-toxic)"
            )
        with gr.Column():
            gr.Markdown(f"**Features:** {data_info['n_features_selected']} (filtered from {data_info['n_features_raw']})")

    with gr.Accordion("📊 Model comparison", open=False):
        metrics_df = pd.DataFrame(pipeline["results"]).sort_values("auc_roc", ascending=False)
        display_cols = ["model", "auc_roc", "accuracy", "f1", "precision", "recall", "specificity"]
        gr.DataFrame(
            metrics_df[display_cols].rename(columns={
                "model": "Model", "auc_roc": "AUC-ROC", "accuracy": "Accuracy",
                "f1": "F1 Score", "precision": "Precision", "recall": "Recall",
                "specificity": "Specificity",
            }),
            datatype=["str", "number", "number", "number", "number", "number", "number"],
        )

    with gr.Accordion("🏆 Feature importance", open=False):
        fi_model = gr.Dropdown(
            choices=[m for m in top3 if hasattr(pipeline["models"][m], "feature_importances_")],
            value=top3[0],
            label="Model",
        )
        fi_plot = gr.BarPlot(
            x="Importance",
            y="Feature",
            title="Top 15 features",
            height=400,
        )

        def update_fi(model_name):
            fi = get_feature_importance(pipeline, model_name, top_n=15)
            if fi is None:
                return gr.update(value=None)
            names, vals = zip(*fi)
            df = pd.DataFrame({"Feature": names, "Importance": vals})
            return gr.update(value=df)

        fi_model.change(fn=update_fi, inputs=fi_model, outputs=fi_plot)

    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown(
            """
            1. Enter a SMILES string (e.g. `CCO` for ethanol)
            2. Featurisation:
               - **Morgan fingerprint** (radius=2, 2048 bits)
               - **217 RDKit descriptors** (logP, MW, TPSA, etc.)
            3. Low-variance features are filtered (same as training pipeline)
            4. The classifier predicts toxicity (log_LC50 < 1.0 → toxic)
            """
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
