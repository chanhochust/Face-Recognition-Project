# database.py
import os
import cv2
import numpy as np
import pickle
from utils import face_mesh, mp_drawing, app, align_face

# Đường dẫn dataset để tạo database
dataset_root = "data/test"
output_mesh_dir = "outputs/meshes"

# Tạo thư mục lưu ảnh có face mesh
os.makedirs(output_mesh_dir, exist_ok=True)

# Tạo database
database = {}
processed_images = 0

# Duyệt từng người
for person_name in os.listdir(dataset_root):
    person_path = os.path.join(dataset_root, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"\nProcessing: {person_name}")
    embeddings = []

    output_person_dir = os.path.join(output_mesh_dir, person_name)
    os.makedirs(output_person_dir, exist_ok=True)

    for img_name in os.listdir(person_path):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping: {img_name}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Vẽ face mesh lên ảnh và lưu
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
            )
            cv2.imwrite(os.path.join(output_person_dir, img_name), annotated_image)

            # Căn chỉnh ảnh khuôn mặt
            aligned = align_face(image, landmarks)
            if aligned is not None:
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                aligned_rgb = cv2.resize(aligned_rgb, (112, 112))
                faces = app.get(aligned_rgb)
                if faces:
                    embeddings.append(faces[0].embedding)
                    processed_images += 1
        else:
            print(f"No face in {img_name}")

    if embeddings:
        mean_embedding = np.mean(np.stack(embeddings), axis=0)
        database[person_name] = mean_embedding
        print(f"Added {person_name}, total embeddings: {len(embeddings)}")

# Lưu database
with open("data/face_database.pkl", "wb") as f:
    pickle.dump(database, f)
print(f"\nSố ảnh được xử lý: {processed_images}")
print("\nDONE. Saved embeddings to data/face_database.pkl")
