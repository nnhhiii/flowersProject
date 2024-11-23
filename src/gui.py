import os
from tkinter import filedialog
import cv2
from PIL import Image
from preprocess import preprocess_image
from predict import predict_flower, gesture_history
import customtkinter as ctk

cap = None
after_event_id_recognition = None

def start_gui(model):
    root = ctk.CTk()
    root.configure(fg_color="white")

    root.title("Nhận dạng cử chỉ tay")

    gesture =  ctk.CTkLabel(root, text="", font=("Verdana", 18, "bold"), text_color="#58CC01")
    gesture.grid(row=1, column=4, columnspan=3, sticky="e", padx=10)

    result_label = ctk.CTkLabel(root, text="")
    result_label.grid(row=8, column=4, columnspan=3)

    title = ctk.CTkLabel(root, text="NHẬN DẠNG HOA", font=("Verdana", 23, "bold"), text_color="#0E3469")
    title.grid(row=0, column=1, columnspan=6, pady=15)

    label = ctk.CTkLabel(root, text="")
    label.grid(row=8, column=2)

    history_label = ctk.CTkLabel(root, text="", wraplength=300, justify="left")
    history_label.grid(row=10, column=4, columnspan=3)

    recognize_btn = ctk.CTkButton(root, text="Nhận diện", text_color="white", corner_radius=50, width=100, height=50, fg_color="#FF86D0", hover_color="#CC6BA7")
    recognize_btn.grid_forget()

    image_label = ctk.CTkLabel(root, text="")



    def show_result(predicted_class, confidence):
        if confidence > 0.5:
            gesture.configure(text=predicted_class)
            result_label.configure(text=f"Cử chỉ được nhận diện: {predicted_class}     Độ chính xác: {confidence:.2f}")
            gesture_history.append(predicted_class)
            history_label.configure(text="Lịch sử nhận diện: " + ", ".join(gesture_history))
        else:
            result_label.configure(text="Không nhận diện được cử chỉ. Vui lòng thực hiện lại.")

    def upload_media():
        file_path = filedialog.askopenfilename(filetypes=[("Image and Video files", "*.jpg *.jpeg *.png *.mp4 *.avi")])
        if not file_path:
            return

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext in [".jpg", ".jpeg", ".png"]:
            # Hiển thị ảnh
            img = Image.open(file_path)
            img_tk = ctk.CTkImage(size=(400, 300), light_image=img)

            # Xóa ảnh cũ nếu có trước khi hiển thị ảnh mới
            image_label.grid_forget()
            image_label.configure(image=img_tk)
            image_label.image = img_tk
            image_label.grid(row=2, column=4, columnspan=3, rowspan=6, pady=15, padx=15)

            def recognize_gesture():
                img_cv = cv2.imread(file_path)
                preprocessed_img = preprocess_image(img_cv)
                predicted_class, confidence = predict_flower(model, preprocessed_img)
                show_result(predicted_class, confidence)

            recognize_btn.configure(command=recognize_gesture)
            recognize_btn.grid(row=9, column=5)
        elif ext in [".mp4", ".avi"]:
            cap = cv2.VideoCapture(file_path)
            frame_skip = 10
            prev_gesture = None  # Initialize prev_gesture here
            def recognize_video():
                nonlocal prev_gesture
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_skip == 0:
                        preprocessed_frame = preprocess_image(frame)
                        predicted_class, confidence = predict_flower(model, preprocessed_frame)

                        if predicted_class != prev_gesture:
                            show_result(predicted_class, confidence)
                            prev_gesture = predicted_class
                    frame_count += 1
                    cv2.imshow("Video Recognition", frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                cap.release()
                cv2.destroyAllWindows()
            recognize_btn.configure(command=recognize_video)
            recognize_btn.grid(row=9, column=5)


    upload_btn = ctk.CTkButton(root, text="Chọn ảnh/video", font=("Arial", 15, "bold"), text_color="white", corner_radius=10, width=180, height=60,
                               fg_color="#4EA3E2", hover_color="#1DC1FF", command=upload_media)
    upload_btn.grid(row=2, column=0, rowspan=2, padx=10, pady=10)

    root.mainloop()
