import cv2
import face_recognition
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

def resize_image(image, max_size):
    height, width = image.shape[:2]
    aspect_ratio = width / height

    if width > height:
        new_width = min(width, max_size)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(height, max_size)
        new_width = int(new_height * aspect_ratio)

    return cv2.resize(image, (new_width, new_height))

def load_known_image():
    global known_image, known_encoding, known_image_label
    filepath = filedialog.askopenfilename(title="Select Known Image")
    if filepath:
        known_image = face_recognition.load_image_file(filepath)
        known_encoding = face_recognition.face_encodings(known_image)[0]

        known_img_resized = resize_image(known_image, 400)
        known_pil_img = Image.fromarray(cv2.cvtColor(known_img_resized, cv2.COLOR_BGR2RGB))
        known_tk_img = ImageTk.PhotoImage(image=known_pil_img)

        known_image_label.config(image=known_tk_img)
        known_image_label.image = known_tk_img

def load_unknown_image():
    global unknown_image, unknown_image_label
    filepath = filedialog.askopenfilename(title="Select Unknown Image")
    if filepath:
        unknown_image = face_recognition.load_image_file(filepath)

        unknown_img_resized = resize_image(unknown_image, 400)
        unknown_pil_img = Image.fromarray(cv2.cvtColor(unknown_img_resized, cv2.COLOR_BGR2RGB))
        unknown_tk_img = ImageTk.PhotoImage(image=unknown_pil_img)

        unknown_image_label.config(image=unknown_tk_img)
        unknown_image_label.image = unknown_tk_img

def recognize_faces():
    global known_image, known_encoding, unknown_image, output_image_label, result_label

    if known_image is None or unknown_image is None:
        result_label.config(text="Please load both images!", fg="black", bg="#ffdb99")
        return

    rgb_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    match_found = False

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        if True in matches:
            match_found = True
            cv2.rectangle(output_image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(output_image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(output_image, "Face Matched", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    output_img_resized = resize_image(output_image, 400)
    output_pil_img = Image.fromarray(cv2.cvtColor(output_img_resized, cv2.COLOR_BGR2RGB))
    output_tk_img = ImageTk.PhotoImage(image=output_pil_img)

    output_image_label.config(image=output_tk_img)
    output_image_label.image = output_tk_img

    if match_found:
        result_label.config(text="Matched", fg="white", bg="green")
    else:
        result_label.config(text="Not Matched", fg="white", bg="red")

root = Tk()
root.title("Face Recognition GUI - Group Image")
root.geometry("1200x600")
root.configure(bg="#f0f0f0")

known_image = None
known_encoding = None
unknown_image = None

known_image_label = Label(root, bg="#f0f0f0")
known_image_label.grid(row=1, column=0, padx=10, pady=10)

unknown_image_label = Label(root, bg="#f0f0f0")
unknown_image_label.grid(row=1, column=1, padx=10, pady=10)

output_image_label = Label(root, bg="#f0f0f0")
output_image_label.grid(row=1, column=2, padx=10, pady=10)

load_known_button = Button(root, text="Load Known Image", font=("Arial", 12), command=load_known_image)
load_known_button.grid(row=0, column=0, padx=20, pady=20)

load_unknown_button = Button(root, text="Load Unknown Image", font=("Arial", 12), command=load_unknown_image)
load_unknown_button.grid(row=0, column=1, padx=20, pady=20)

recognize_button = Button(root, text="Recognize Faces", font=("Arial", 12), command=recognize_faces)
recognize_button.grid(row=0, column=2, padx=20, pady=20)

result_label = Label(root, text="", font=("Arial", 16), width=20, height=2)
result_label.grid(row=3, column=1, pady=20)

Label(root, text="Known Image", font=("Arial", 14), bg="#f0f0f0").grid(row=2, column=0, padx=10, pady=10)
Label(root, text="Unknown Image", font=("Arial", 14), bg="#f0f0f0").grid(row=2, column=1, padx=10, pady=10)
Label(root, text="Output Image", font=("Arial", 14), bg="#f0f0f0").grid(row=2, column=2, padx=10, pady=10)

root.mainloop()
