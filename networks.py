from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from keras.models import Model, Sequential

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
N_EMBEDDING = 128  # Size of the embedding layer.
N_LABELS = 41  # Size of output layer.


def get_n1():
    input_shape = (NUM_FRAMES, NUM_BANDS, 1)
    img_input = Input(shape=input_shape)
    model = Sequential()

    # Block 1
    model.add(Conv2D(8, (3, 3), padding='same', name='conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))
    model.add(Dropout(0.3))

    # Block 2
    #model.add(Conv2D(16, (3, 3), padding='same', name='conv2'))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D((2, 2), strides=(1, 1), name='pool2'))

    # Block fc
    model.add(Flatten(name='flatten'))
    #model.add(Dense(41, name='fc1_1'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    return model


def get_vggish():
    input_shape = (NUM_FRAMES, NUM_BANDS, 1)
    img_input = Input(shape=input_shape)
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))

    # Block fc
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1_1'))
    model.add(Dense(4096, activation='relu', name='fc1_2'))
    model.add(Dense(N_EMBEDDING, activation='relu', name='fc2'))

    return model
