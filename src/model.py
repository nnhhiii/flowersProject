import os

from keras.src.applications.densenet import DenseNet121
from keras.src.applications.efficientnet import EfficientNetB7
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.applications.resnet import ResNet50
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, AveragePooling2D, Dense, \
    GlobalAveragePooling2D, MaxPooling2D, Concatenate, Flatten, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential


# 1. Hàm tạo ImageDataGenerator cho train và validation
def create_generators(train_data_dir, validation_data_dir, target_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator

# 2. Mô hình CNN thủ công (Simple CNN)
def create_simple_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)

    # Lớp Conv2D đầu tiên với 32 bộ lọc, kích thước 3x3 và hàm kích hoạt ReLU
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    x = BatchNormalization()(x)  # Chuẩn hóa batch
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Lớp Conv2D thứ hai với 64 bộ lọc và hàm kích hoạt ReLU
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Khối Residual
    def residual_block(x, filters, kernel_size=(3, 3)):
        shortcut = x
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])  # Thêm phần residual
        return x

    # Áp dụng khối Residual
    x = residual_block(x, 128)
    x = residual_block(x, 256)

    # Lớp Conv2D với số lượng bộ lọc lớn hơn và Pooling
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Lớp Dropout để tránh overfitting
    x = Dropout(0.5)(x)

    # Sử dụng Global Average Pooling để giảm chiều dữ liệu
    x = GlobalAveragePooling2D()(x)

    # Lớp Fully Connected với 1024 nơ-ron và hàm kích hoạt ReLU
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Lớp Dropout để tránh overfitting

    # Lớp phân loại với số lượng lớp đầu ra (num_classes)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Xây dựng mô hình
    model = Model(inputs=input_layer, outputs=predictions)

    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 3. Mô hình VGG16 thủ công (Deep CNN)
def create_vgg16_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)

    # Block 1: 2 lớp Conv + MaxPooling
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2: 2 lớp Conv + MaxPooling
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3: 3 lớp Conv + MaxPooling
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 4: 3 lớp Conv + MaxPooling
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 5: 3 lớp Conv + MaxPooling
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flattening và Fully Connected layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)  # Lớp fully connected 4096 nơ-ron
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', kernel_initializer='he_normal')(x)  # Lớp fully connected 4096 nơ-ron
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Lớp đầu ra với số lớp cần phân loại

    # Xây dựng mô hình
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


    # base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    #
    # # Không cho các tầng convolutional này học lại
    # for layer in base_model.layers:
    #     layer.trainable = False
    #
    # # Thêm các lớp Fully Connected (FC) cho bài toán mới
    # x = base_model.output
    # x = Flatten()(x)
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # predictions = Dense(num_classes, activation='softmax')(x)
    #
    # # Xây dựng mô hình
    # model = Model(inputs=base_model.input, outputs=predictions)
    #
    # # Biên dịch mô hình
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # return model


# 4. Mô hình ResNet thủ công (Residual blocks)
def create_resnet_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)

    # Residual block
    def residual_block(x, filters, kernel_size=(3, 3)):
        shortcut = x
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = Add()([x, shortcut])  # Thêm phần residual
        return x

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')(input_layer)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 256)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 5. Mô hình Inception thủ công (Inception block)
def create_inception_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)

    # Khối Inception
    def inception_block(x, filters_1x1, filters_3x3, filters_5x5, filters_pool):
        # Lớp Convolution 1x1
        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
        conv_1x1 = BatchNormalization()(conv_1x1)  # Chuẩn hóa batch

        # Lớp Convolution 3x3
        conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(x)
        conv_3x3 = BatchNormalization()(conv_3x3)  # Chuẩn hóa batch

        # Lớp Convolution 5x5
        conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(x)
        conv_5x5 = BatchNormalization()(conv_5x5)  # Chuẩn hóa batch

        # MaxPooling kết hợp với lớp Convolution 1x1
        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_1x1 = Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(pool)
        pool_1x1 = BatchNormalization()(pool_1x1)  # Chuẩn hóa batch

        # Kết hợp tất cả các đầu ra của các lớp convolution và pooling lại
        return Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_1x1])

    # Áp dụng khối Inception đầu tiên
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Khối Inception 1
    x = inception_block(x, 64, 128, 32, 32)

    # Khối Inception 2
    x = inception_block(x, 128, 256, 64, 64)

    # Khối Inception 3
    x = inception_block(x, 256, 512, 128, 128)

    # Sử dụng Global Average Pooling để giảm chiều dữ liệu
    x = AveragePooling2D(pool_size=(8, 8))(x)

    # Flatten và các lớp Fully Connected
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Lớp Dropout để tránh overfitting
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Lớp Dropout để tránh overfitting
    predictions = Dense(num_classes, activation='softmax')(x)  # Lớp phân loại đầu ra

    # Xây dựng mô hình
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  # Biên dịch mô hình

    return model



# 6. Mô hình DenseNet thủ công (Dense block)
def create_densenet_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)

    # Hàm tạo Dense Block
    def dense_block(x, num_layers, growth_rate):
        for _ in range(num_layers):
            # Bottleneck Layer (1x1 Convolution để giảm chiều dữ liệu)
            bn1 = BatchNormalization()(x)
            relu1 = ReLU()(bn1)
            bottleneck = Conv2D(4 * growth_rate, (1, 1), padding='same', activation='relu')(relu1)

            # 3x3 Convolution để tăng chiều dữ liệu
            bn2 = BatchNormalization()(bottleneck)
            relu2 = ReLU()(bn2)
            conv = Conv2D(growth_rate, (3, 3), padding='same', activation='relu')(relu2)

            # Nối đầu ra với đầu vào ban đầu (Dense Connection)
            x = Concatenate()([x, conv])
        return x

    # Hàm tạo Transition Layer
    def transition_layer(x, reduction):
        bn = BatchNormalization()(x)
        relu = ReLU()(bn)
        filters = int(x.shape[-1] * reduction)  # Giảm số lượng kênh
        x = Conv2D(filters, (1, 1), padding='same', activation='relu')(relu)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        return x

    # Layer đầu tiên (Stem Layer)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # Dense Block 1
    x = dense_block(x, num_layers=6, growth_rate=32)
    x = transition_layer(x, reduction=0.5)  # Transition Layer 1

    # Dense Block 2
    x = dense_block(x, num_layers=12, growth_rate=32)
    x = transition_layer(x, reduction=0.5)  # Transition Layer 2

    # Dense Block 3
    x = dense_block(x, num_layers=24, growth_rate=32)
    x = transition_layer(x, reduction=0.5)  # Transition Layer 3

    # Dense Block 4
    x = dense_block(x, num_layers=16, growth_rate=32)

    # Global Average Pooling và lớp Fully Connected
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Xây dựng mô hình
    model = Model(inputs=input_layer, outputs=predictions)

    # Biên dịch mô hình
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 7. Hàm huấn luyện mô hình
def train_model(model, train_generator, validation_generator, epochs=10, early_stopping_patience=3):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=early_stopping_patience,
                                   restore_best_weights=True)

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )
