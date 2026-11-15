"""
SuperKart Sales Forecasting API
Endpoints:
  GET  /health          -> service health check
  POST /v1/predict      -> single-product online inference
  POST /v1/predictbatch -> batch inference from a CSV upload
"""
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_PATH = "superkart_model.joblib"
_model = None


def get_model():
    """Lazily load the serialized model on first request."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Align raw incoming data with the schema the model was trained on."""
    df = df.copy()
    if "Product_Sugar_Content" in df.columns:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    return df


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None}), 200


@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "No JSON body provided"}), 400
        df = _normalize(pd.DataFrame([payload]))
        prediction = get_model().predict(df)[0]
        return jsonify({"prediction": float(prediction)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing file upload under key 'file'"}), 400
        df = _normalize(pd.read_csv(request.files["file"]))
        preds = get_model().predict(df)
        return jsonify({str(i): float(p) for i, p in enumerate(preds)}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
