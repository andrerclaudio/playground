# Build-in modules
import logging
import os
from datetime import timedelta

import cv2
from keras import backend as k
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
# Added modules
from keras.preprocessing.image import ImageDataGenerator
from pytictoc import TicToc

# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)


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


def center_crop(img, dim):
    """

    """
    width, height = img.shape[1], img.shape[0]

    crop_width = (dim[0] if dim[0] < img.shape[1] else img.shape[1])
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)

    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]

    return crop_img


def resize(image, proportion=None, size=None, gray=False, crop=None):
    """

    """

    resized = None

    if crop:
        dim = (crop, crop)
        image = center_crop(image, dim)

    if proportion:
        width = int(image.shape[1] * proportion / 100)
        height = int(image.shape[0] * proportion / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    if size:
        dim = (size, size)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    if gray:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    logger.debug(f'Resized dimensions: {resized.shape}')
    return resized


def generate_dataset(path):
    """ """
    # define a video file
    vid = cv2.VideoCapture(path)
    idx = 1
    counting = 1

    train_test_proportion = 4 / 1

    while vid.isOpened():

        ret, frame = vid.read()
        if ret:
            frame = resize(frame,
                           proportion=GeneralSettings().IMG_PROPORTION,
                           gray=GeneralSettings().gray,
                           crop=GeneralSettings().crop)
            rel = idx % train_test_proportion

            if rel:
                cv2.imwrite('./pics/train/off/off_green_train_{}.jpg'.format(counting), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 100])
                counting += 1
            else:
                test_idx = int(idx / train_test_proportion)
                cv2.imwrite('./pics/test/off/off_green_test_{}.jpg'.format(test_idx), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 100])

            idx += 1
        else:
            break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()


class GeneralSettings(object):
    """
    General settings initialization
    """

    def __init__(self):
        self.LABELS = ['on', 'off']
        self.IMG_SIZE = 224
        self.IMG_PROPORTION = 100
        self.gray = False
        self.crop = self.IMG_SIZE
        self.TRAIN_DATA_PATH = './pics/train'
        self.VALIDATION_DATA_PATH = './pics/test'
        self.NUMBER_TRAIN_SAMPLES = 30
        self.NUMBER_VALIDATION_SAMPLES = 10
        self.EPOCHS = 15
        self.BATCH_SIZE = 2


def application():
    """" All application has its initialization from here """
    logger.info('Main application is running!')

    tm = ElapsedTime()
    image = False
    cwd = os.getcwd()

    # generate_dataset(cwd + '/pics/off.mp4', )

    try:

        if k.image_data_format() == 'channels_first':
            input_shape = (3, GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE)
        else:
            input_shape = (GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE, 3)

        model = Sequential()
        model.add(Conv2D(32, (2, 2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        train_data_generator = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_data_generator = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data_generator.flow_from_directory(
            GeneralSettings().TRAIN_DATA_PATH,
            target_size=(GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE),
            batch_size=GeneralSettings().BATCH_SIZE,
            class_mode='binary')

        validation_generator = test_data_generator.flow_from_directory(
            GeneralSettings().VALIDATION_DATA_PATH,
            target_size=(GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE),
            batch_size=GeneralSettings().BATCH_SIZE,
            class_mode='binary')

        model.fit(
            train_generator,
            steps_per_epoch=GeneralSettings().NUMBER_TRAIN_SAMPLES // GeneralSettings().BATCH_SIZE,
            epochs=GeneralSettings().EPOCHS,
            validation_data=validation_generator,
            validation_steps=GeneralSettings().NUMBER_VALIDATION_SAMPLES // GeneralSettings().BATCH_SIZE)

        # model.save_weights('model_saved.h5')

        # model = load_model('model_saved.h5')

        # define a video capture object
        vid = cv2.VideoCapture(cwd + '/pics/cut.mp4')

        while vid.isOpened():
            # Capture the video frame by frame
            ret, frame = vid.read()

            # if ret:
            # image = resize(frame,
            #                proportion=GeneralSettings().IMG_PROPORTION,
            #                gray=GeneralSettings().gray,
            #                crop=GeneralSettings().crop
            #                )
            #
            # img = np.array(image)
            # img = img / 255.0
            # img = img.reshape(1, GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE, 3)
            #
            # label = model.predict(img)
            # classes_x = np.argmax(label, axis=1)
            # value = classes_x.item()
            #
            # if value == 0:
            #     label = GeneralSettings().LABELS[1]
            # else:
            #     label = GeneralSettings().LABELS[0]
            #
            # cv2.putText(frame,
            #             str(label),
            #             (50, 50),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 255, 255),
            #             2,
            #             cv2.LINE_4)

            cv2.imshow("Frame", frame)

        # After the loop release the cap object
        vid.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.exception(e, exc_info=False)

    finally:
        tm.elapsed()
        return image
