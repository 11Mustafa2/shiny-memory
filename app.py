from flask import Flask, request, jsonify, Response
import joblib
import numpy as np
import time

# ----------------- LOAD MODEL -----------------
app = Flask(__name__)

# Load bundle
bundle = joblib.load("bee_comfort_final.pkl")

# Select best model
best_model_name = bundle["best_model_name"]

if best_model_name == "CNN":
    model = bundle["cnn"]
    keras_model = True
elif best_model_name == "LSTM":
    model = bundle["lstm"]
    keras_model = True
elif best_model_name == "SVM":
    model = bundle["svm"]
    keras_model = False

scaler_X = bundle["scaler_X"]
scaler_y = bundle["scaler_y"]
features = bundle["features"]

print("Best model loaded:", best_model_name)

# IMPORTANT: SAME ORDER AS TRAINING
feature_cols = features


# ----------------- HELPER: predict correctly -----------------
def make_prediction(x):
    if keras_model:
        pred = model.predict(x)
        return float(pred[0][0])
    else:
        pred = model.predict(x)
        return float(pred[0])


# ----------------- /predict -----------------
@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json(silent=True)

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features'"}), 400

    x = np.array(data["features"]).reshape(1, -1)

    # scale input
    x_scaled = scaler_X.transform(x)

    pred = make_prediction(x_scaled)

    # unscale output
    pred_unscaled = scaler_y.inverse_transform([[pred]])[0][0]

    return jsonify({"prediction": float(pred_unscaled)})


# ----------------- /stream -----------------
@app.route("/stream")
def stream():
    def generate():
        while True:
            # random input for testing
            rand_x = np.random.rand(len(features)).reshape(1, -1)


            rand_x_scaled = scaler_X.transform(rand_x)

            pred = make_prediction(rand_x_scaled)
            pred_unscaled = scaler_y.inverse_transform([[pred]])[0][0]

            yield f"data: {float(pred_unscaled)}\n\n"
            time.sleep(1)

    return Response(generate(), mimetype="text/event-stream")


# ----------------- RUN SERVER -----------------
if __name__ == "__main__":
    print("🚀 Flask server running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", port=5050)
