const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const result = document.getElementById("result");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;"
    })[character]);
}

function readImageAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Could not read the selected MRI image."));
        reader.readAsDataURL(file);
    });
}


// ==========================================
// CHECK IMAGE FILE
// ==========================================

function isBrainMriFile(file) {
    if (!file) return false;

    const validMimeTypes = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ];

    const validExtensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ];

    const filename = file.name.toLowerCase();

    const isValidMime =
        validMimeTypes.includes(file.type.toLowerCase());

    const hasValidExtension =
        validExtensions.some(
            (ext) => filename.endsWith(ext)
        );

    return isValidMime || hasValidExtension;
}


// ==========================================
// IMAGE PREVIEW
// ==========================================

function updatePreview(file) {

    if (!file) {

        imagePreview.innerHTML =
            "<span>No image selected yet</span>";

        return;
    }

    const reader = new FileReader();

    reader.onload = (event) => {

        imagePreview.innerHTML = `
            <img
                src="${event.target.result}"
                alt="MRI preview"
            >
        `;
    };

    reader.readAsDataURL(file);
}


// ==========================================
// SHOW RESULT
// ==========================================

function setResult(message, type = "info") {

    result.className = `result-box ${type}`;

    result.innerHTML = message;
}


// ==========================================
// IMAGE SELECTED
// ==========================================

imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (!file) {

        setResult(
            "Upload a scan to receive a prediction."
        );

        return;
    }


    if (!isBrainMriFile(file)) {

        imageInput.value = "";

        imagePreview.innerHTML =
            "<span>No image selected yet</span>";

        setResult(
            "Error: Please upload a valid image file (PNG or JPG only).",
            "error"
        );

        return;
    }


    updatePreview(file);

    setResult(`
        Selected file:
        <strong>${file.name}</strong>
    `);
});


// ==========================================
// RESET
// ==========================================

resetBtn.addEventListener("click", () => {

    imageInput.value = "";

    updatePreview(null);

    setResult(
        "Upload a scan to receive a prediction."
    );
});


// ==========================================
// PREDICT TUMOR
// ==========================================

// async function predictTumor() {

//     if (
//         !imageInput.files ||
//         imageInput.files.length === 0
//     ) {

//         setResult(
//             "Please select an MRI image first.",
//             "error"
//         );

//         return;
//     }


//     const imageFile =
//         imageInput.files[0];


//     if (!isBrainMriFile(imageFile)) {

//         setResult(
//             "Error: Please upload a valid image file (PNG or JPG only).",
//             "error"
//         );

//         return;
//     }


//     // Create form data
//     const formData = new FormData();

//     formData.append(
//         "image",
//         imageFile
//     );


//     // Loading message
//     setResult(
//         "Analyzing MRI scan... Please wait.",
//         "loading"
//     );

//     analyzeBtn.disabled = true;


//     try {

//         // ==================================
//         // SEND IMAGE TO FLASK
//         // ==================================

//         const response = await fetch(
//             "http://127.0.0.1:5000/predict",
//             {
//                 method: "POST",
//                 body: formData
//             }
//         );


//         // Get JSON response
//         const data =
//             await response.json();


//         console.log(
//             "Backend Response:",
//             data
//         );


//         // ==================================
//         // ERROR CHECK
//         // ==================================

//         if (!response.ok || data.error) {

//             setResult(
//                 data.error ||
//                 "Prediction failed. Please try again.",
//                 "error"
//             );

//             return;
//         }


//         // ==================================
//         // GET BACKEND RESULT
//         // ==================================

//         const prediction =
//             data.prediction ||
//             "Prediction unavailable";


//         const confidence =
//             data.confidence;


//         const filename =
//             data.filename ||
//             imageFile.name;


//         // ==================================
//         // SHOW RESULT
//         // ==================================

//         setResult(`
//             <div class="result-title">
//                 ${prediction}
//             </div>

//             <div class="result-meta">
//                 Confidence:
//                 <strong>${confidence}%</strong>
//             </div>

//             <div class="result-meta">
//                 File:
//                 <strong>${filename}</strong>
//             </div>
//         `);


//     } catch (error) {

//         console.error(
//             "Prediction error:",
//             error
//         );

//         setResult(
//             "Unable to connect to the Flask backend. Start the backend server first.",
//             "error"
//         );


//     } finally {

//         analyzeBtn.disabled = false;

//     }
// }

async function predictTumor() {

    if (!imageInput.files || imageInput.files.length === 0) {
        setResult("Please select an MRI image first.", "error");
        return;
    }

    const imageFile = imageInput.files[0];

    if (!isBrainMriFile(imageFile)) {
        setResult(
            "Error: Please upload a valid image file (PNG or JPG only).",
            "error"
        );
        return;
    }

    const formData = new FormData();

    formData.append("image", imageFile);

    setResult(
        "Analyzing MRI scan... Please wait.",
        "loading"
    );

    analyzeBtn.disabled = true;

    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        console.log("Backend Response:", data);

        if (!response.ok || data.error) {

            setResult(
                data.error || "Prediction failed. Please try again.",
                "error"
            );

            return;
        }

        const confidence = Number(data.class_probability ?? data.confidence);
        const prediction = data.prediction || "Prediction unavailable";
        const filename = data.filename || imageFile.name;
        const message = data.message || "MRI scan analyzed successfully.";
        const predictedClass = data.predicted_class || data.tumor_type || prediction;
        const saliency = data.saliency;
        const saliencyStatus = data.saliency_status ||
            "No tumor attention map was generated for this result.";
        const normalizedClass = predictedClass.toLowerCase().replace(/[\s_-]/g, "");
        const formattedClass = ["notumor", "normal"].includes(normalizedClass)
            ? "No Tumor"
            : predictedClass
                .replaceAll("_", " ")
                .replace(/\b\w/g, (character) => character.toUpperCase());

        if (!Number.isFinite(confidence)) {
            setResult(
                "Prediction failed because the backend returned an invalid confidence value.",
                "error"
            );
            return;
        }

        const resultType = data.status === "uncertain"
            ? "warning"
            : prediction === "Tumor detected"
                ? "danger"
                : "success";
        const analysisStatus = data.status === "uncertain"
            ? "Uncertain — low model confidence"
            : "Success";
        const uploadedImageUrl = await readImageAsDataUrl(imageFile);

        setResult(`
            <div class="analysis-result-grid">
                <section class="prediction-details" aria-labelledby="predictionHeading">
                    <h4 id="predictionHeading">Prediction Information</h4>
                    <div class="prediction-label">Prediction</div>
                    <div class="result-title">${escapeHtml(formattedClass)}</div>
                    <dl class="result-fields">
                        <div>
                            <dt>Confidence</dt>
                            <dd>${confidence.toFixed(2)}%</dd>
                        </div>
                        <div>
                            <dt>File</dt>
                            <dd>${escapeHtml(filename)}</dd>
                        </div>
                        <div>
                            <dt>Brain MRI screening</dt>
                            <dd>${escapeHtml(data.brain_scan_screening || "passed")}</dd>
                        </div>
                        <div>
                            <dt>Analysis status</dt>
                            <dd>${escapeHtml(analysisStatus)}</dd>
                        </div>
                    </dl>
                    <p class="analysis-message">${escapeHtml(message)}</p>
                </section>

                <section class="analysis-image-panel" aria-labelledby="analyzedImageHeading">
                    <h4 id="analyzedImageHeading">Analyzed MRI</h4>
                    <div class="mri-visualization">
                        <img
                            class="result-image"
                            src="${uploadedImageUrl}"
                            alt="Original uploaded brain MRI scan"
                        >
                        ${saliency ? `
                        <img
                            class="saliency-overlay"
                            src="${saliency}"
                            alt=""
                            aria-hidden="true"
                        >
                        ` : ""}
                    </div>
                    <strong class="attention-label">Model Attention Region</strong>
                    <small class="saliency-description">
                        ${saliency
                            ? "Highlighted region represents the image areas that contributed to the model prediction. It is not an exact tumor boundary."
                            : escapeHtml(saliencyStatus)}
                    </small>
                </section>
            </div>
        `, resultType);

    } catch (error) {

        console.error("Frontend error:", error);

        setResult(
            error instanceof Error && error.message === "Could not read the selected MRI image."
                ? error.message
                : "Unable to connect to the Flask backend. Start the backend server first.",
            "error"
        );

    } finally {

        analyzeBtn.disabled = false;

    }
}
// ==========================================
// ANALYZE BUTTON
// ==========================================

analyzeBtn.addEventListener(
    "click",
    predictTumor
);