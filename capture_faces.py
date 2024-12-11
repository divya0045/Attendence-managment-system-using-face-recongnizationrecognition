import cv2
import os

def capture_faces(user_id, user_name):
    folder_path = f"dataset/{user_id}_{user_name}"
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            file_path = os.path.join(folder_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) == ord("q") or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} face images for User ID: {user_id}, Name: {user_name}")
