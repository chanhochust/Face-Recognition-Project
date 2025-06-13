# recognition.py
import cv2
import numpy as np
import pickle
from utils import face_mesh, app, align_face, cv2_imshow

# Load database
with open("data/face_database.pkl", "rb") as f:
    database = pickle.load(f)

# Hàm tính cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Hàm chuyển cosine similarity thành xác suất
def cosine_to_sigmoid_prob(sim, scale=10):
    return 1 / (1 + np.exp(-scale * (sim - 0.5)))*100

# Thay bằng ảnh tùy thích
image_path = 'data/sample_images/test2.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Không thể đọc ảnh! Vui lòng kiểm tra đường dẫn.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Lấy landmark khuôn mặt
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        aligned, (x1, y1, x2, y2) = align_face(image, landmarks)
        if aligned is not None:
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            faces = app.get(aligned_rgb)
            if faces:
                embedding = faces[0].embedding

                # So sánh với database và tính xác suất
                similarities = []
                for name, db_embed in database.items():
                    sim = cosine_similarity(embedding, db_embed)
                    prob = cosine_to_sigmoid_prob(sim)
                    similarities.append((name, sim, prob))

                # Lấy người có độ tương đồng cao nhất
                name, sim, prob = max(similarities, key=lambda x: x[1])
                threshold = 0.6
                if sim >= threshold:
                    label = f"{name} ({prob:.2f}%)"
                else:
                    label = "Unknown"

                # Vẽ khung + tên
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)

                # Hiển thị ảnh
                cv2_imshow(image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Không phát hiện khuôn mặt trong ảnh đã căn chỉnh!")
        else:
            print("Không thể căn chỉnh khuôn mặt!")
    else:
        print("Không phát hiện khuôn mặt trong ảnh!")
