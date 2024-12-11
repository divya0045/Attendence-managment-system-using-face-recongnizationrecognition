import tkinter as tk
from tkinter import messagebox, scrolledtext
from capture_faces import capture_faces
from train_model import train_face_recognition_model
from recognize_face import recognize_face_and_mark_attendance

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("700x600")
        self.face_recognizer = None
        self.label_dict = None

        # Title Label
        title_label = tk.Label(root, text="Face Recognition Attendance System", font=("Helvetica", 18, "bold"))
        title_label.pack(pady=20)

        # User Input Fields
        input_frame = tk.Frame(root)
        input_frame.pack(pady=20)

        user_id_label = tk.Label(input_frame, text="Enter User ID:", font=("Helvetica", 12))
        user_id_label.pack(side=tk.LEFT, padx=5)
        self.user_id_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=15)
        self.user_id_entry.pack(side=tk.LEFT, padx=5)

        user_name_label = tk.Label(input_frame, text="Enter User Name:", font=("Helvetica", 12))
        user_name_label.pack(side=tk.LEFT, padx=5)
        self.user_name_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=15)
        self.user_name_entry.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=40)

        self.capture_button = tk.Button(button_frame, text="Capture Face", command=self.capture_face, width=15, font=("Helvetica", 12))
        self.capture_button.pack(side=tk.LEFT, padx=10)

        self.train_button = tk.Button(button_frame, text="Train Model", command=self.train_model, width=15, font=("Helvetica", 12))
        self.train_button.pack(side=tk.LEFT, padx=10)

        self.recognize_button = tk.Button(button_frame, text="Mark Attendance", command=self.recognize_face, width=15, font=("Helvetica", 12))
        self.recognize_button.pack(side=tk.LEFT, padx=10)

        # Show Records Button and Display Area
        self.show_records_button = tk.Button(root, text="Show Attendance Records", command=self.show_records, width=25, font=("Helvetica", 12))
        self.show_records_button.pack(pady=20)

        # Scrolled Text Area to show attendance records
        self.record_area = scrolledtext.ScrolledText(root, width=80, height=15, font=("Helvetica", 12))
        self.record_area.pack(pady=10)

    def capture_face(self):
        user_id = self.user_id_entry.get()
        user_name = self.user_name_entry.get()
        if user_id and user_name:
            capture_faces(user_id, user_name)
            messagebox.showinfo("Capture Complete", f"Captured face data for User ID: {user_id}, Name: {user_name}")
        else:
            messagebox.showwarning("Input Error", "Please enter both User ID and User Name before capturing face data.")

    def train_model(self):
        self.face_recognizer, self.label_dict = train_face_recognition_model()
        messagebox.showinfo("Training Complete", "Model training completed successfully!")

    def recognize_face(self):
        if self.face_recognizer:
            recognize_face_and_mark_attendance(self.face_recognizer, self.label_dict)
        else:
            messagebox.showwarning("Error", "Model not trained yet! Train the model first.")

    def show_records(self):
        """Show attendance records in a scrolled text area."""
        try:
            with open("attendance.csv", "r") as file:
                records = file.readlines()
                if not records:
                    messagebox.showinfo("No Records", "No attendance records found.")
                else:
                    self.record_area.delete(1.0, tk.END)  # Clear any previous records
                    for record in records:
                        self.record_area.insert(tk.END, record)
        except FileNotFoundError:
            messagebox.showwarning("File Not Found", "Attendance file not found. Please capture some attendance first.")

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()
