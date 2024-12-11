import cv2
import numpy as np
import os

def train_face_recognition_model():
    face_data = []
    labels = []
    label_dict = {}
    current_label = 0

    # Read dataset
    for user_id in os.listdir("dataset"):
        user_path = f"dataset/{user_id}"
        for face_image in os.listdir(user_path):
            img = cv2.imread(f"{user_path}/{face_image}", cv2.IMREAD_GRAYSCALE)
            face_data.append(img)
            labels.append(current_label)

        label_dict[current_label] = user_id
        current_label += 1

    # Train recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_data, np.array(labels))
    face_recognizer.save("face_model.yml")

    print("Model trained and saved as face_model.yml")
    return face_recognizer, label_dict
