import cv2
from datetime import datetime

marked_attendance = set()

def recognize_face_and_mark_attendance(face_recognizer, label_dict):
    global marked_attendance
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)

            if confidence < 100:
                user_info = label_dict[label]  # Retrieve "UserID_UserName"
                if user_info not in marked_attendance:
                    mark_attendance(user_info)
                    marked_attendance.add(user_info)

                # Display User ID and Name
                cv2.putText(frame, f"{user_info}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Display "Unknown" for unrecognized faces
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(user_info):
    with open("attendance.csv", "a") as file:
        file.write(f"{user_info},{datetime.now()}\n")
    print(f"Attendance marked for: {user_info}")
