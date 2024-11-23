import os

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
    model = Sequential()

    # Thêm lớp Conv2D đầu tiên với 32 bộ lọc, kích thước 3x3 và hàm kích hoạt ReLU
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Thêm lớp Conv2D thứ hai với 64 bộ lọc và hàm kích hoạt ReLU
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Thêm lớp Conv2D thứ ba với 128 bộ lọc và hàm kích hoạt ReLU
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Chuyển từ 3D (ảnh) sang 1D (vector) để vào lớp Dense
    model.add(Flatten())

    # Lớp Dense đầu tiên với 512 nơ-ron và hàm kích hoạt ReLU
    model.add(Dense(512, activation='relu'))

    # Lớp Dropout để tránh overfitting
    model.add(Dropout(0.5))

    # Lớp Dense cuối cùng với số lượng lớp đầu ra tương ứng với số lớp phân loại (num_classes)
    model.add(Dense(num_classes, activation='softmax'))

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

    def inception_block(x, filters):
        branch1x1 = Conv2D(filters, (1, 1), activation='relu', padding='same')(x)

        branch3x3 = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

        branch5x5 = Conv2D(filters, (5, 5), activation='relu', padding='same')(x)

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = Conv2D(filters, (1, 1), activation='relu', padding='same')(branch_pool)

        x = Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])
        return x

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = inception_block(x, 128)
    x = inception_block(x, 256)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 6. Mô hình DenseNet thủ công (Dense block)
def create_densenet_model(input_shape=(224, 224, 3), num_classes=5):
    input_layer = Input(shape=input_shape)

    # Dense block
    def dense_block(x, filters):
        x1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(x1)
        return Concatenate()([x, x1, x2])

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = dense_block(x, 64)
    x = dense_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 7. Hàm huấn luyện mô hình
def train_model(model, train_generator, validation_generator, epochs=10, early_stopping_patience=3):
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )
