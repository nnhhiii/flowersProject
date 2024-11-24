import numpy as np
import tensorflow as tf

flower_history = []

def predict_flower(model, preprocessed_image):
    # Ds
    class_names = ["Hoa lan", "Hoa huệ", "Hoa hồng", "Hoa hướng dương", "Hoa sen"]

    # Dự đoán cử chỉ từ hình ảnh đã được tiền xử lý
    predictions = model.predict(preprocessed_image)

    # Lấy chỉ số của lớp có xác suất cao nhất
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
    confidence = np.max(predictions)

    # Lấy tên lớp dựa trên chỉ số
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name, confidence
