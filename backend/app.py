from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops

app = FastAPI()

# =========================
# MODEL 1: Asli vs Bukan
# =========================
interpreter_asli = tf.lite.Interpreter(model_path="model_asli_bukan.tflite")
interpreter_asli.allocate_tensors()
input_details_asli = interpreter_asli.get_input_details()
output_details_asli = interpreter_asli.get_output_details()
norm_asli = np.load("norm_asli_bukan.npz")
X_min_asli = norm_asli["X_min"]
X_max_asli = norm_asli["X_max"]

# =========================
# MODEL 2: Motif (3 Kelas)
# =========================
interpreter_motif = tf.lite.Interpreter(model_path="model_motif.tflite")
interpreter_motif.allocate_tensors()
input_details_motif = interpreter_motif.get_input_details()
output_details_motif = interpreter_motif.get_output_details()
norm_motif = np.load("norm_motif.npz")
X_min_motif = norm_motif["X_min"]
X_max_motif = norm_motif["X_max"]

# =========================
# Fungsi Normalisasi & GLCM
# =========================
def normalize(features, X_min, X_max):
    return (features - X_min) / (X_max - X_min + 1e-8)

def extract_glcm(img):
    glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    features = [graycoprops(glcm, p)[0, 0] for p in props]
    return np.array(features, dtype=np.float32)

# =========================
# Endpoint Gabungan
# =========================
@app.post("/predict_final")
async def predict_final(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))

        # === Ekstrak fitur GLCM
        features = extract_glcm(image)

        # === Tahap 1: Asli vs Bukan
        norm_feat_asli = normalize(features, X_min_asli, X_max_asli).reshape(1, -1).astype(np.float32)
        interpreter_asli.set_tensor(input_details_asli[0]['index'], norm_feat_asli)
        interpreter_asli.invoke()
        output_asli = interpreter_asli.get_tensor(output_details_asli[0]['index'])
        class_idx_asli = int(np.argmax(output_asli))
        label_asli = "Asli" if class_idx_asli == 0 else "Bukan_Asli"
        confidence_asli = float(output_asli[0][class_idx_asli])

        # === Jika bukan asli, tidak lanjut motif
        if label_asli == "Bukan_Asli":
            return JSONResponse(content={
                "asli_bukan": label_asli,
                "confidence": confidence_asli
            })

        # === Tahap 2: Klasifikasi Motif
        norm_feat_motif = normalize(features, X_min_motif, X_max_motif).reshape(1, -1).astype(np.float32)
        interpreter_motif.set_tensor(input_details_motif[0]['index'], norm_feat_motif)
        interpreter_motif.invoke()
        output_motif = interpreter_motif.get_tensor(output_details_motif[0]['index'])
        class_idx_motif = int(np.argmax(output_motif))
        confidence_motif = float(output_motif[0][class_idx_motif])
        label_motif = ["Amanuban", "Amanatun", "Molo"][class_idx_motif]

        # === Gabungkan hasil
        return JSONResponse(content={
            "asli_bukan": label_asli,
            "confidence_asli": confidence_asli,
            "motif": label_motif,
            "confidence_motif": confidence_motif
        })

    except Exception as e:
        print("Error saat prediksi akhir:", e)
        return JSONResponse(status_code=500, content={"error": "Terjadi kesalahan saat proses prediksi akhir"})
