from pathlib import Path
import sys
import json
import base64
from io import BytesIO

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.models import load_model


# ==========================================
# PATH SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import hybrid_model  # Registers the custom layers used by the hybrid model.

MULTICLASS_MODEL_PATH = BASE_DIR / "models" / "gcn_transformer_brain_tumor.keras"
LEGACY_MODEL_PATH = BASE_DIR / "brain_tumor_model.keras"
MODEL_PATH = (
    MULTICLASS_MODEL_PATH
    if MULTICLASS_MODEL_PATH.exists()
    else LEGACY_MODEL_PATH
)
CLASS_NAMES_PATH = BASE_DIR / "models" / "gcn_transformer_brain_tumor_classes.json"


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

MODEL = load_model(
    str(MODEL_PATH),
    custom_objects={
        "GraphConvolution": hybrid_model.GraphConvolution,
        "TokenPositionEmbedding": hybrid_model.TokenPositionEmbedding,
    },
    compile=False,
)
if CLASS_NAMES_PATH.exists():
    with CLASS_NAMES_PATH.open(encoding="utf-8") as class_file:
        CLASS_NAMES = json.load(class_file)
else:
    training_dir = BASE_DIR / "data" / "Training"
    if training_dir.exists():
        CLASS_NAMES = sorted(
            item.name for item in training_dir.iterdir() if item.is_dir()
        )
    else:
        CLASS_NAMES = ["no_tumor", "tumor"]

IMG_SIZE = (128, 128)
MIN_CLASS_CONFIDENCE = 0.60
MAX_COLOR_PIXEL_FRACTION = 0.03

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg"
}


print("====================================")
print("Brain Tumor Detection Backend")
print("====================================")
print(f"Model loaded from: {MODEL_PATH}")
print("Server is ready!")
print("====================================")


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message": "Brain Tumor Detection Backend is Running!",
        "status": "healthy"
    })


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image_file):

    image = ImageOps.exif_transpose(Image.open(image_file)).convert("RGB")

    image = ImageOps.contain(
        image,
        IMG_SIZE,
        method=Image.Resampling.BILINEAR,
    )
    padded_image = Image.new("RGB", IMG_SIZE, color=(0, 0, 0))
    padded_image.paste(
        image,
        ((IMG_SIZE[0] - image.width) // 2, (IMG_SIZE[1] - image.height) // 2),
    )

    image_array = np.array(
        padded_image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Model already contains Rescaling(1/255)
    # so DON'T divide by 255 here.

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def screen_brain_mri_input(image_file):
    image_file.stream.seek(0)
    image = ImageOps.exif_transpose(Image.open(image_file))
    aspect_ratio = image.width / image.height
    if not 0.75 <= aspect_ratio <= 1.33:
        image_file.stream.seek(0)
        return (
            "The image proportions do not match the brain MRI scans supported by "
            "this model. Please upload a brain MRI image, not a screen capture."
        )

    image = image.convert("RGB")
    rgb_sample = np.asarray(image.resize(IMG_SIZE), dtype=np.int16)
    channel_spread = rgb_sample.max(axis=2) - rgb_sample.min(axis=2)
    colored_pixel_fraction = float(np.mean(channel_spread > 20))
    image_file.stream.seek(0)

    if colored_pixel_fraction > MAX_COLOR_PIXEL_FRACTION:
        return (
            "This image contains substantial color and does not look like a "
            "grayscale brain MRI. Please upload a brain MRI scan."
        )

    grayscale = np.asarray(
        ImageOps.grayscale(image).resize(IMG_SIZE),
        dtype=np.float32,
    )
    foreground_threshold = max(
        12.0,
        float(np.percentile(grayscale, 70)) * 0.35,
    )
    foreground = grayscale > foreground_threshold
    components, component_count = ndimage.label(foreground)
    if component_count == 0:
        return "No brain-like scan area was found. Please upload a brain MRI scan."

    component_sizes = np.bincount(components.ravel())
    component_sizes[0] = 0
    largest_component = components == int(np.argmax(component_sizes))
    rows, columns = np.where(largest_component)
    if not len(columns):
        return "No brain-like scan area was found. Please upload a brain MRI scan."

    occupancy = float(largest_component.mean())
    width_fraction = float((columns.max() - columns.min() + 1) / IMG_SIZE[1])
    height_fraction = float((rows.max() - rows.min() + 1) / IMG_SIZE[0])
    center_x = float(columns.mean() / IMG_SIZE[1])
    center_y = float(rows.mean() / IMG_SIZE[0])
    bilateral_similarity = float(
        1.0
        - np.logical_xor(
            largest_component[:, : IMG_SIZE[1] // 2],
            np.fliplr(largest_component[:, IMG_SIZE[1] // 2 :]),
        ).mean()
    )

    looks_like_brain_slice = (
        0.15 <= occupancy <= 0.95
        and width_fraction >= 0.5
        and height_fraction >= 0.55
        and 0.42 <= center_x <= 0.58
        and 0.30 <= center_y <= 0.75
        and bilateral_similarity >= 0.65
    )
    image_file.stream.seek(0)
    if not looks_like_brain_slice:
        return (
            "The image does not have the expected centered brain-scan structure. "
            "Please upload a clear brain MRI scan."
        )
    return None


def create_saliency_overlay(
    image_array,
    output_size,
    class_index=None,
    binary_tumor_score=False,
):
    """Build a transparent, class-specific input-gradient attribution overlay."""
    input_tensor = tf.convert_to_tensor(image_array)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        scores = MODEL(input_tensor, training=False)
        target_score = (
            tf.reduce_sum(scores[:, 0])
            if binary_tumor_score
            else tf.reduce_sum(scores[:, class_index])
        )

    gradients = tape.gradient(target_score, input_tensor)
    if gradients is None:
        raise RuntimeError("Could not calculate input gradients for saliency.")

    image_values = image_array[0] / 255.0
    saliency = np.mean(
        np.abs(gradients.numpy()[0] * image_values),
        axis=-1,
    )
    saliency = ndimage.gaussian_filter(saliency, sigma=2)
    if not np.isfinite(saliency).all() or float(saliency.max()) <= 0:
        raise RuntimeError("The model did not produce a usable saliency map.")

    grayscale = np.mean(image_array[0], axis=-1)
    foreground_threshold = max(12.0, float(np.percentile(grayscale, 70)) * 0.35)
    foreground = grayscale > foreground_threshold
    labels, foreground_count = ndimage.label(foreground)
    if foreground_count == 0:
        raise RuntimeError("Could not identify the brain foreground for localization.")
    foreground_sizes = np.bincount(labels.ravel())
    foreground_sizes[0] = 0
    brain_region = labels == int(np.argmax(foreground_sizes))
    brain_region = ndimage.binary_erosion(brain_region, iterations=2)
    if not np.any(brain_region):
        raise RuntimeError("Could not isolate the brain foreground for localization.")

    saliency[~brain_region] = 0
    region_values = saliency[brain_region]
    if region_values.size == 0 or float(region_values.max()) <= 0:
        raise RuntimeError("The model did not produce saliency inside the brain region.")

    high_threshold = float(np.quantile(region_values, 0.90))
    low_threshold = float(np.quantile(region_values, 0.65))
    if high_threshold <= low_threshold:
        high_threshold = float(region_values.max())
        low_threshold = float(region_values.min())
    normalized = np.clip(
        (saliency - low_threshold) / max(high_threshold - low_threshold, 1e-12),
        0,
        1,
    )
    alpha = np.where(brain_region, normalized * 175, 0).astype(np.uint8)
    overlay_array = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    overlay_array[..., 0] = 255
    overlay_array[..., 1] = 42
    overlay_array[..., 2] = 56
    overlay_array[..., 3] = alpha
    overlay = Image.fromarray(overlay_array, mode="RGBA").resize(
        output_size,
        Image.Resampling.BILINEAR,
    )
    output = BytesIO()
    overlay.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ==========================================
# PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check image
    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    image = request.files["image"]


    # Check filename
    if image.filename == "":

        return jsonify({
            "error": "No file selected"
        }), 400


    # Check file type
    if image.mimetype not in ALLOWED_TYPES:

        return jsonify({
            "error": "Please upload a valid image file (PNG or JPG)."
        }), 400

    try:
        validation_error = screen_brain_mri_input(image)
        if validation_error:
            return jsonify({
                "error": validation_error,
                "status": "rejected"
            }), 422
    except Exception as exc:
        print(f"Image validation error: {exc}")
        return jsonify({
            "error": "The uploaded image could not be validated. Please try another MRI image.",
            "status": "error"
        }), 400


    try:

        # -----------------------------
        # Preprocess image
        # -----------------------------

        processed = preprocess_image(image)


        # -----------------------------
        # Model prediction
        # -----------------------------

        prediction_values = MODEL.predict(
            processed,
            verbose=0
        )[0]
        saliency = None
        saliency_status = "Not generated because the model did not predict a tumor."

        if len(prediction_values) > 1:
            probabilities = np.asarray(prediction_values, dtype=np.float32)
            if len(CLASS_NAMES) != len(probabilities):
                raise ValueError(
                    "Model output classes do not match the configured class names: "
                    f"{len(probabilities)} outputs vs {len(CLASS_NAMES)} names."
                )
            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            is_no_tumor = predicted_class.lower().replace("_", "").replace("-", "") in {
                "notumor",
                "normal",
            }
            confidence = float(probabilities[predicted_index])
            tumor_type = (
                None if is_no_tumor else predicted_class
            )
            result = (
                "No tumor detected"
                if is_no_tumor
                else "Tumor detected"
            )
            no_tumor_index = next(
                (
                    index
                    for index, class_name in enumerate(CLASS_NAMES)
                    if class_name.lower().replace("_", "").replace("-", "")
                    in {"notumor", "normal"}
                ),
                None,
            )
            probability = (
                float(1.0 - probabilities[no_tumor_index])
                if no_tumor_index is not None
                else float(probabilities[predicted_index])
            )
            tumor_type_status = "Predicted by the multiclass GCN + Transformer model."
            if not is_no_tumor:
                try:
                    image.stream.seek(0)
                    source_image = ImageOps.exif_transpose(
                        Image.open(image)
                    )
                    output_scale = min(
                        1.0,
                        1024 / max(source_image.size),
                    )
                    output_size = tuple(
                        max(1, round(dimension * output_scale))
                        for dimension in source_image.size
                    )
                    saliency = create_saliency_overlay(
                        processed,
                        output_size=output_size,
                        class_index=predicted_index,
                    )
                    saliency_status = (
                        "Highlighted region represents image areas that contributed "
                        "to this model prediction. It is not a tumor segmentation."
                    )
                except (RuntimeError, ValueError) as exc:
                    print(f"Saliency map unavailable: {exc}")
                    saliency_status = (
                        "Prediction returned, but a saliency map could not be generated."
                    )
        else:
            probability = float(prediction_values[0])
            result = "Tumor detected" if probability >= 0.5 else "No tumor detected"
            confidence = probability if probability >= 0.5 else 1.0 - probability
            predicted_class = "tumor" if probability >= 0.5 else "no_tumor"
            tumor_type = None
            tumor_type_status = (
                "Unavailable: the loaded model is binary and has no tumor-type classes."
            )
            if probability >= 0.5:
                try:
                    image.stream.seek(0)
                    source_image = ImageOps.exif_transpose(
                        Image.open(image)
                    )
                    output_scale = min(
                        1.0,
                        1024 / max(source_image.size),
                    )
                    output_size = tuple(
                        max(1, round(dimension * output_scale))
                        for dimension in source_image.size
                    )
                    saliency = create_saliency_overlay(
                        processed,
                        output_size=output_size,
                        binary_tumor_score=True,
                    )
                    saliency_status = (
                        "Highlighted region represents image areas that contributed "
                        "to this model prediction. It is not a tumor segmentation."
                    )
                except (RuntimeError, ValueError) as exc:
                    print(f"Saliency map unavailable: {exc}")
                    saliency_status = (
                        "Prediction returned, but a saliency map could not be generated."
                    )

        is_uncertain = confidence < MIN_CLASS_CONFIDENCE
        analysis_status = "uncertain" if is_uncertain else "success"
        analysis_message = (
            (
                f"The model's best-match class is {predicted_class}, but its confidence "
                f"({confidence * 100:.2f}%) is below the "
                f"{MIN_CLASS_CONFIDENCE * 100:.0f}% reliability threshold. Treat this "
                "result as inconclusive, not as a diagnosis."
            )
            if is_uncertain
            else "MRI scan analyzed successfully."
        )

        # Convert to percentage
        confidence_percent = round(
            confidence * 100,
            2
        )


        print(
            f"Image: {image.filename} | "
            f"Probability: {probability:.4f} | "
            f"Result: {result} | "
            f"Confidence: {confidence_percent}%"
        )


        # -----------------------------
        # Send result to frontend
        # -----------------------------

        return jsonify({
            "message": analysis_message,
            "brain_scan_screening": "passed",
            "filename": image.filename,
            "prediction": result,
            "predicted_class": predicted_class,
            "class_probability": confidence_percent,
            "tumor_type": tumor_type,
            "tumor_type_status": tumor_type_status,
            "saliency": saliency,
            "saliency_status": saliency_status,
            "saliency_method": "Class-specific input-times-gradient",
            "probability": round(probability, 6),
            "tumor_probability": round(probability * 100, 2),
            "no_tumor_probability": round((1.0 - probability) * 100, 2),
            "confidence": confidence_percent,
            "confidence_decimal": round(confidence, 6),
            "status": analysis_status
        })


    except Exception as exc:

        print(
            f"Prediction error: {exc}"
        )

        return jsonify({
            "error": "Unable to process the uploaded image."
        }), 400


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )