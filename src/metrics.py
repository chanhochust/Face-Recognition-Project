# metrics.py
import os
import cv2
import numpy as np
import pickle
from utils import face_mesh, app, align_face
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Load database
with open("data/face_database.pkl", "rb") as f:
    database = pickle.load(f)

# Đường dẫn thư mục ảnh để đánh giá
test_dir = "data/test_images"

true_labels = []
pred_labels = []
similarity_vectors = []

# Duyệt từng ảnh
for person_name in os.listdir(test_dir):
    person_path = os.path.join(test_dir, person_name)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            aligned = align_face(image, landmarks)
            if aligned is None:
                continue

            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            aligned_rgb = cv2.resize(aligned_rgb, (112, 112))
            faces = app.get(aligned_rgb)
            if not faces:
                continue

            embedding = faces[0].embedding

            # Tính cosine similarity với tất cả người trong database
            similarities = []
            for db_embed in database.values():
                sim = np.dot(embedding, db_embed) / (np.linalg.norm(embedding) * np.linalg.norm(db_embed))
                similarities.append(sim)

            pred_idx = int(np.argmax(similarities))
            pred_name = list(database.keys())[pred_idx]

            true_labels.append(person_name)
            pred_labels.append(pred_name)
            similarity_vectors.append(similarities)

# Confusion Matrix
all_labels = sorted(set(true_labels + pred_labels))
cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Face Recognition")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_cosine.png")
plt.show()

# AUC-ROC (One-vs-All)
label_indices = {name: idx for idx, name in enumerate(database.keys())}
y_true_bin = label_binarize([label_indices[label] for label in true_labels], classes=list(range(len(database))))
scores = np.array(similarity_vectors)

plt.figure(figsize=(10, 8))
for i, name in enumerate(database.keys()):
    y_true = y_true_bin[:, i]
    y_score = scores[:, i]
    if len(np.unique(y_true)) < 2:
        print(f"Bỏ qua {name} vì không đủ dữ liệu để vẽ ROC.")
        continue
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curve_cosine.png")
plt.show()
