# utils.py
import cv2
import numpy as np
import mediapipe as mp
from insightface.app import FaceAnalysis
from google.colab.patches import cv2_imshow

# Khởi tạo model
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
app = FaceAnalysis(name='buffalo_s')
app.prepare(ctx_id=-1, det_size=(320, 320))

# Hàm căn chỉnh khuôn mặt
def align_face(image, landmarks, output_size=(112, 112)):
    h, w = image.shape[:2]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    eyes_center = ((left_eye.x * w + right_eye.x * w) * 0.5,
                   (left_eye.y * h + right_eye.y * h) * 0.5)
    dY = right_eye.y * h - left_eye.y * h
    dX = right_eye.x * w - left_eye.x * w
    angle = np.degrees(np.arctan2(dY, dX))
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    face_width = abs(right_eye.x * w - left_eye.x * w) * 2.0
    face_height = face_width * 1.0
    x_center = eyes_center[0]
    y_center = eyes_center[1]
    x1 = int(x_center - face_width / 2)
    x2 = int(x_center + face_width / 2)
    y1 = int(y_center - face_height / 2)
    y2 = int(y_center + face_height / 2)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    face_img = aligned_image[y1:y2, x1:x2]
    if face_img.size == 0:
        print("Cannot crop face region!")
        return None
    face_img = cv2.resize(face_img, output_size, interpolation=cv2.INTER_AREA)
    print("Cropped region (x1, y1, x2, y2):", (x1, y1, x2, y2))
    return face_img
