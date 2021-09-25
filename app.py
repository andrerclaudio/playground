# Build-in modules
import logging
from datetime import timedelta

# Added modules
import matplotlib

matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from pytictoc import TicToc
# import seaborn as sns
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report, confusion_matrix
# import tensorflow as tf
import cv2
import os
import numpy as np

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)

LABELS = ['on', 'off']
IMG_SIZE = 224


class ElapsedTime(object):
    """
    Measure the elapsed time between Tic and Toc
    """

    def __init__(self):
        self.t = TicToc()
        self.t.tic()

    def elapsed(self):
        _elapsed = self.t.tocvalue()
        d = timedelta(seconds=_elapsed)
        logger.info('< {} >'.format(d))


def resize(image, proportion=0.3):
    """ """
    width = int(image.shape[1] * proportion / 100)
    height = int(image.shape[0] * proportion / 100)
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    logger.debug(f'Resized dimensions: {resized.shape}')
    return resized


def get_data(data_dir):
    """ """
    data = []

    for label in LABELS:

        path = os.path.join(data_dir, label)
        class_num = LABELS.index(label)

        for img in os.listdir(path):

            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])

            except Exception as e:
                logger.exception(e, exc_info=False)

    return np.array(data, dtype=object)


def application():
    """" All application has its initialization from here """
    logger.info('Main application is running!')

    tm = ElapsedTime()
    image = False

    # define a video file
    vid = cv2.VideoCapture('./pics/cut.mp4')
    idx = 1

    while vid.isOpened():
        ret, frame = vid.read()
        # Convert to RGB format
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = resize(img_rgb, proportion=50)

        cv2.imwrite('./pics/train/on/on_green_train_{}.jpg'.format(idx), img_rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
        idx += 1

        # Display the resulting frame
        # cv2.imshow('frame', img_rgb)
        # cv2.waitKey()
        # break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()

    try:
        pass
        # train = get_data('./pics/train')
        # val = get_data('./pics/test')
        #
        # x_train = []
        # y_train = []
        # x_val = []
        # y_val = []
        #
        # for feature, label in train:
        #     x_train.append(feature)
        #     y_train.append(label)
        #
        # for feature, label in val:
        #     x_val.append(feature)
        #     y_val.append(label)
        #
        # # Normalize the data
        # x_train = np.array(x_train) / 255
        # x_val = np.array(x_val) / 255
        #
        # x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        # y_train = np.array(y_train)
        #
        # x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        # y_val = np.array(y_val)
        #
        # data_generator = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        #     zoom_range=0.2,  # Randomly zoom image
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images
        #
        # data_generator.fit(x_train)
        #
        # model = Sequential()
        # model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3)))
        # model.add(MaxPool2D())
        # model.add(Conv2D(32, 3, padding="same", activation="relu"))
        # model.add(MaxPool2D())
        # model.add(Conv2D(64, 3, padding="same", activation="relu"))
        # model.add(MaxPool2D())
        # model.add(Dropout(0.4))
        # model.add(Flatten())
        # model.add(Dense(128, activation="relu"))
        # model.add(Dense(2, activation="softmax"))
        # model.summary()
        # opt = Adam(lr=0.000001)
        # model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #               metrics=['accuracy'])
        #
        # history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))
        #
        # predictions = model.predict_classes(x_val)
        # predictions = predictions.reshape(1, -1)[0]
        # logger.info(classification_report(y_val, predictions, target_names=['Rugby (Class 0)', 'Soccer (Class 1)']))

    except Exception as e:
        logger.exception(e, exc_info=False)

    finally:
        tm.elapsed()
        return image
