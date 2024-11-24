import os

from gui import start_gui
from tensorflow.keras.models import load_model


def main():
    models = {
        "vgg16": load_model('../models/vgg16_model.keras'),
        "resnet": load_model('../models/resnet_model.keras'),
        "inception": load_model('../models/inception_model.keras'),
        "densenet": load_model('../models/densenet_model.keras'),
        "cnn": load_model('../models/cnn_model.keras'),
    }
    # Bắt đầu giao diện người dùng
    start_gui(models)


if __name__ == "__main__":
    # Kiểm tra nếu mô hình chưa tồn tại thì huấn luyện, nếu không thì tải mô hình đã huấn luyện
    from model import *

    train_data_dir = '../data/train'
    validation_data_dir = '../data/validation'
    train_generator, validation_generator = create_generators(train_data_dir, validation_data_dir)

    # Tạo và huấn luyện mô hình VGG16 thủ công
    vgg16_model_path = '../models/vgg16_model.keras'
    if not os.path.exists(vgg16_model_path):
        model_vgg16 = create_vgg16_model(input_shape=(224, 224, 3), num_classes=5)
        train_model(model_vgg16, train_generator, validation_generator, epochs=10, early_stopping_patience=3)
        model_vgg16.save(vgg16_model_path)
    else:
        print(f'Model VGG16 already exists at {vgg16_model_path}, skipping training.')


    # Tạo và huấn luyện mô hình ResNet thủ công
    resnet_model_path = '../models/resnet_model.keras'
    if not os.path.exists(resnet_model_path):
        model_resnet = create_resnet_model(input_shape=(224, 224, 3), num_classes=5)
        train_model(model_resnet, train_generator, validation_generator, epochs=10, early_stopping_patience=3)
        model_resnet.save(resnet_model_path)
    else:
        print(f'Model ResNet already exists at {resnet_model_path}, skipping training.')

    # Tạo và huấn luyện mô hình Inception thủ công
    inception_model_path = '../models/inception_model.keras'
    if not os.path.exists(inception_model_path):
        model_inception = create_inception_model(input_shape=(224, 224, 3), num_classes=5)
        train_model(model_inception, train_generator, validation_generator)
        model_inception.save(inception_model_path)
    else:
        print(f'Model Inception already exists at {inception_model_path}, skipping training.')

    # Tạo và huấn luyện mô hình DenseNet thủ công
    densenet_model_path = '../models/densenet_model.keras'
    if not os.path.exists(densenet_model_path):
        model_densenet = create_densenet_model(input_shape=(224, 224, 3), num_classes=5)
        train_model(model_densenet, train_generator, validation_generator)
        model_densenet.save(densenet_model_path)
    else:
        print(f'Model DenseNet already exists at {densenet_model_path}, skipping training.')

    cnn_model_path = '../models/cnn_model.keras'
    if not os.path.exists(cnn_model_path):
        model_cnn = create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=5)
        train_model(model_cnn, train_generator, validation_generator)
        model_cnn.save(cnn_model_path)
    else:
        print(f'Model CNN already exists at {cnn_model_path}, skipping training.')

    main()
