# preprocessing.py
import os
import cv2
import numpy as np
from utils import face_mesh, mp_drawing, drawing_spec, app, align_face, cv2_imshow

# Danh sách sample
image_paths = ['data/sample_images/003.jpg', 'data/sample_images/004.jpg']

# Xử lý từng ảnh
embeddings = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read {image_path}! Check the file path.")
        continue

    print(f"Processing {os.path.basename(image_path)} - Original image shape:", image.shape)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        # Tạo bản sao của ảnh gốc để vẽ Face Mesh
        image_with_mesh = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            # Vẽ Face Mesh lên ảnh gốc
            mp_drawing.draw_landmarks(
                image=image_with_mesh,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
        print(f"Image {os.path.basename(image_path)} with Face Mesh:")
        cv2_imshow(image_with_mesh)

        # Căn chỉnh và tính embedding
        for face_landmarks in results.multi_face_landmarks:
            aligned_face = align_face(image, face_landmarks.landmark)
            if aligned_face is not None:
                print(f"Aligned {os.path.basename(image_path)} face image (shape):", aligned_face.shape)
                cv2_imshow(aligned_face)
                aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                if aligned_face_rgb.shape[:2] != (112, 112):
                    print(f"Resizing {os.path.basename(image_path)} to 112x112...")
                    aligned_face_rgb = cv2.resize(aligned_face_rgb, (112, 112), interpolation=cv2.INTER_AREA)
                print(f"Input {os.path.basename(image_path)} shape for InsightFace:", aligned_face_rgb.shape)
                try:
                    faces = app.get(aligned_face_rgb)
                    if len(faces) > 0:
                        embedding = faces[0].embedding
                        embeddings.append((os.path.basename(image_path), embedding))
                        print(f"Feature embedding shape for {os.path.basename(image_path)}:", embedding.shape)
                        # Lưu embedding thành file .npy
                        np.save(f"data/{os.path.splitext(os.path.basename(image_path))[0]}_embedding.npy", embedding)
                        print(f"Saved embedding to data/{os.path.splitext(os.path.basename(image_path))[0]}_embedding.npy")
                    else:
                        print(f"No face detected in aligned {os.path.basename(image_path)}!")
                except Exception as e:
                    print(f"Error during face detection for {os.path.basename(image_path)}:", str(e))
    else:
        print(f"No face detected in {os.path.basename(image_path)}!")

# So sánh hai ảnh
if len(embeddings) == 2:
    img1_name, emb1 = embeddings[0]
    img2_name, emb2 = embeddings[1]
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"\nCosine Similarity between {img1_name} and {img2_name}: {similarity:.4f}")
    if similarity > 0.5:
        print("The two faces are likely the same person.")
    else:
        print("The two faces are likely different people.")
else:
    print("Could not compare images due to insufficient embeddings!")

face_mesh.close()
