import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.datasets import cifar10
from tensorflow.compat.v1.keras.preprocessing.image import ImageDataGenerator


def build_resnet18(input_shape=(32, 32, 3), num_classes=10):
    input_tensor = tf.keras.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    x = _build_resnet_block(x, 64, 2, block_name='block1')
    x = _build_resnet_block(x, 128, 2, block_name='block2')
    x = _build_resnet_block(x, 256, 2, block_name='block3')
    x = _build_resnet_block(x, 512, 2, block_name='block4')

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_tensor, outputs=x, name='resnet18')
    return model


def _build_resnet_block(x, filters, blocks, block_name):
    for i in range(blocks):
        shortcut = x
        stride = 1
        if i == 0:
            stride = 2  # downsample on first iteration

        y = layers.Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)

        y = layers.Conv2D(filters, (3, 3), padding='same')(y)
        y = layers.BatchNormalization()(y)

        # Shortcut connection
        if stride != 1 or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=(stride, stride), padding='valid')(x)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([y, shortcut])
        x = layers.Activation('relu')(x)

    return x


# Create and compile the model
model = build_resnet18()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to be between 0 and 1

# Build ResNet-18 model
model = build_resnet18(input_shape=(32, 32, 3), num_classes=10)

# Data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 64
epochs = 20

model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc*100:.2f}%")
