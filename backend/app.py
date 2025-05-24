from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops

app = FastAPI()

# =========================
# LOAD MODEL 1: Asli vs Bukan
# =========================
interpreter_asli = tf.lite.Interpreter(model_path="model_asli_bukan.tflite")
interpreter_asli.allocate_tensors()
input_details_asli = interpreter_asli.get_input_details()
output_details_asli = interpreter_asli.get_output_details()
norm_asli = np.load("norm_asli_bukan.npz")
X_min_asli = norm_asli["X_min"]
X_max_asli = norm_asli["X_max"]

# =========================
# LOAD MODEL 2: Motif
# =========================
interpreter_motif = tf.lite.Interpreter(model_path="model_motif.tflite")
interpreter_motif.allocate_tensors()
input_details_motif = interpreter_motif.get_input_details()
output_details_motif = interpreter_motif.get_output_details()
norm_motif = np.load("norm_motif.npz")
X_min_motif = norm_motif["X_min"]
X_max_motif = norm_motif["X_max"]

# =========================
# Utility: Normalisasi dan GLCM
# =========================
def normalize(features, X_min, X_max):
    return (features - X_min) / (X_max - X_min + 1e-8)

def extract_glcm(img):
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    return np.array([graycoprops(glcm, p)[0, 0] for p in props], dtype=np.float32)

def is_image_file(file: UploadFile):
    return file.content_type in ["image/jpeg", "image/png", "image/jpg"]

# =========================
# Endpoint: Prediksi GLCM + TFLite
# =========================
@app.post("/predict_final")
async def predict_final(file: UploadFile = File(...)):
    if not is_image_file(file):
        return JSONResponse(status_code=400, content={"error": "File harus berupa gambar (jpg, jpeg, png)"})
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Gagal membaca gambar."})

        image = cv2.resize(image, (224, 224))
        features = extract_glcm(image)

        # ======= Model Asli/Bukan =======
        norm_feat_asli = normalize(features, X_min_asli, X_max_asli).reshape(1, -1).astype(np.float32)
        interpreter_asli.set_tensor(input_details_asli[0]['index'], norm_feat_asli)
        interpreter_asli.invoke()
        output_asli = interpreter_asli.get_tensor(output_details_asli[0]['index'])
        class_idx_asli = int(np.argmax(output_asli))
        label_asli = "Asli" if class_idx_asli == 0 else "Bukan_Asli"
        confidence_asli = float(output_asli[0][class_idx_asli])

        # ======= Jika Bukan Asli: return early
        if label_asli == "Bukan_Asli":
            return JSONResponse(content={
                "asli_bukan": label_asli,
                "confidence_asli": confidence_asli
            })

        # ======= Model Motif =======
        try:
            norm_feat_motif = normalize(features, X_min_motif, X_max_motif).reshape(1, -1).astype(np.float32)
            interpreter_motif.set_tensor(input_details_motif[0]['index'], norm_feat_motif)
            interpreter_motif.invoke()
            output_motif = interpreter_motif.get_tensor(output_details_motif[0]['index'])
            class_idx_motif = int(np.argmax(output_motif))
            confidence_motif = float(output_motif[0][class_idx_motif])
            label_motif = ["Amanuban", "Amanatun", "Molo"][class_idx_motif]

            return JSONResponse(content={
                "asli_bukan": label_asli,
                "confidence_asli": confidence_asli,
                "motif": label_motif,
                "confidence_motif": confidence_motif
            })

        except Exception as motif_error:
            return JSONResponse(status_code=500, content={
                "error": "Prediksi motif gagal meskipun kain asli.",
                "asli_bukan": label_asli,
                "confidence_asli": confidence_asli,
                "detail": str(motif_error)
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": "Terjadi kesalahan saat proses prediksi akhir",
            "detail": str(e)
        })
