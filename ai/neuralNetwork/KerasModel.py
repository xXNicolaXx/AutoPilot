import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters
import cv2
import pandas as pd
import ntpath
import random
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
datadir = "../data/csv"
imagedir = "../data/images/"
columns = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]
data = pd.read_csv(os.path.join(datadir, "driving_log.csv"), names=columns)
pd.set_option("display.max_colwidth", -1)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
num_bins = 25
samples_per_bin = 1500


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


def show_all_data():
    print(data.describe())
    print(data.head())


def cut_image_path():
    data["center"] = imagedir + data["center"].apply(path_leaf)
    data["left"] = imagedir + data["left"].apply(path_leaf)
    data["right"] = imagedir + data["right"].apply(path_leaf)


def show_initial_steering_data():
    i_hist, i_bins = np.histogram(data["steering"], num_bins)
    center = (i_bins[:-1] + i_bins[1:]) * 0.5
    plt.bar(center, i_hist, width=0.05)
    plt.title("Steering data")
    plt.xlabel("Steering angle")
    plt.ylabel("Number of data")
    plt.plot((np.min(data["steering"]), np.max(data["steering"])), (samples_per_bin, samples_per_bin))
    plt.show()


show_initial_steering_data()


hist, bins = np.histogram(data["steering"], num_bins)
remove_list = []
for i in range(num_bins):
    list_ = []
    for j in range(len(data["steering"])):
        if bins[i] <= data["steering"][j] <= bins[i + 1]:
            list_.append(j)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

data.drop(data.index[remove_list], inplace=True)


def show_modified_steering_data():
    m_hist, _ = np.histogram(data['steering'], num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(center, m_hist, width=0.05)
    plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
    plt.show()


show_all_data()
show_modified_steering_data()


def load_data():
    # TODO: use all three angle
    image_path = data[["center", "left", "right"]].values
    steering = data["steering"].values

    # Check data linearity
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
    # axes[0].set_title('Training set')
    # axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
    # axes[1].set_title('Validation set')
    # image = image_path[10]
    # image_ = mpimg.imread(image)
    # imageP = plt.imshow(image_)
    # plt.show()
    # plt.show()

    return train_test_split(image_path, steering, test_size=0.2, random_state=0)


def zoom_image(image):
    zoom = augmenters.Affine(scale=(1, 1.3))
    return zoom.augment_image(image)


def pan_image(image):
    pan = augmenters.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    return pan.augment_image(image)


def image_brightness(image):
    brightness = augmenters.Multiply((0.2, 1.2))
    return brightness.augment_image(image)


def flip_image(image, steering_angle):
    # We need to "flip" also the steering angle as the image as flipped
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle


def augment_image(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan_image(image)
    if np.random.rand() < 0.5:
        image = zoom_image(image)
    if np.random.rand() < 0.5:
        image = image_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flip_image(image, steering_angle)

    return image, steering_angle


def image_preprocess(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))  # Image input size of the Nvidia model architecture
    image = image / 255

    return image


def batch_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_image = []
        batch_steering = []
        for _ in range(batch_size):
            index = random.randint(0, len(image_paths) - 1)

            center, left, right = image_paths[index]

            steering_angle = steering_angles[index]
            if is_training:

                # TODO: try to use different angle images

                random_image = np.random.choice(3)
                if random_image == 0:
                    image, steering_angle = augment_image(center, steering_angle)
                elif random_image == 1:
                    image, steering_angle = augment_image(left, steering_angle + 0.2)
                else:
                    image, steering_angle = augment_image(right, steering_angle - 0.2)
                # image, steering_angle = augment_image(image, steering_angle)
            else:
                image = mpimg.imread(center)
                steering_angle = steering_angle

            image = image_preprocess(image)
            batch_image.append(image)
            batch_steering.append(steering_angle)
        yield (np.asarray(batch_image), np.asarray(batch_steering))


def nvidia_model():
    """
        NVIDIA model used
        Image normalization to avoid saturation and make gradients work better.
        Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Drop out (0.5)
        Fully connected: neurons: 100, activation: ELU
        Fully connected: neurons: 50, activation: ELU
        Fully connected: neurons: 10, activation: ELU
        Fully connected: neurons: 1 (output)
        # the convolution layers are meant to handle feature engineering
        the fully connected layer for predicting the steering angle.
        dropout avoids overfitting
        ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
        """
    model = Sequential()
    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="elu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="elu"))
    model.add(Flatten())

    model.add(Dense(units=100, activation="elu"))
    model.add(Dense(units=50, activation="elu"))
    model.add(Dense(units=10, activation="elu"))
    model.add(Dense(units=1))

    optimizer = Adam(lr=1e-4)
    model.compile(loss="mse", optimizer=optimizer)

    return model


nvidia_model = nvidia_model()
print(nvidia_model.summary())

start_time = time.clock()
cut_image_path()
X_train, X_valid, y_train, y_valid = load_data()
history = nvidia_model.fit_generator(batch_generator(X_train, y_train, 32, True),
                                     steps_per_epoch=len(X_train),
                                     epochs=5,
                                     validation_data=batch_generator(X_valid, y_valid, 32, False),
                                     validation_steps=200,
                                     verbose=1,
                                     shuffle=1)
print("--- trained in %s seconds ---" % (time.clock() - start_time))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
nvidia_model.save('model1.h5')
