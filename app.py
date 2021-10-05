# Build-in modules
import logging
import os
import sys
from datetime import timedelta

# Added modules
import cv2
import numpy as np
from imutils import paths
from pytictoc import TicToc
from skimage import feature
from skimage.metrics import structural_similarity as ssim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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


def center_crop(img, w, h):
    """

    """
    width, height = img.shape[1], img.shape[0]

    crop_width = w if w < width else width
    crop_height = h if h < height else height

    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)

    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]

    return crop_img


def gray(image):
    """

    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize(image, proportion=None, size=None):
    """

    """

    resized = None

    if proportion:
        width = int(image.shape[1] * proportion / 100)
        height = int(image.shape[0] * proportion / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    if size:
        dim = (size, size)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

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
            frame = center_crop(frame, GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE)
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
        self.n_trials = 10
        self.n_estimators = 1000
        self.IMG_SIZE = 224
        self.IMG_PROPORTION = 100


def path_router():
    """ Application parameter initializer """

    # Store current working directory
    path = os.path.abspath('')
    # Append current directory to the python path
    sys.path.append(path)

    # Wave
    training_path = path + '/pics/train'
    testing_path = path + '/pics/test'
    return training_path, testing_path


def quantify_image(image):
    # compute the histogram of oriented gradients feature vector for the input image
    features = feature.hog(image,
                           orientations=9,
                           pixels_per_cell=(10, 10),
                           cells_per_block=(2, 2),
                           transform_sqrt=True,
                           block_norm="L1")

    # return the feature vector
    return features


def load_split(path):
    # grab the list of images in the input directory, then initialises the list of data
    # and class labels
    image_paths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for image_path in image_paths:
        # extract the class label from the filename
        label = image_path.split(os.path.sep)[-2]

        # load the input image, convert it to grayscale, and resize it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(image_path)
        image = gray(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.resize(image, (200, 200))

        # threshold the image such that the drawing appears as white on a black background
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # quantify the image
        features = quantify_image(image)

        # update the data and labels lists, respectively
        data.append(features)
        labels.append(label)

    # return the data and labels
    return np.array(data), np.array(labels)


def application():
    """" All application has its initialization from here """

    logger.info('Main application is running!')
    tm = ElapsedTime()
    model = None

    # Store current working directory
    path = os.path.abspath('')
    # generate_dataset(path + '/pics/off.mp4', )

    try:
        training_path, testing_path = path_router()

        # loading the training and testing data
        logger.info('Loading data ...')

        (train_x, train_y) = load_split(training_path)
        (test_x, test_y) = load_split(testing_path)

        # encode the labels as integers
        le = LabelEncoder()
        train_y = le.fit_transform(train_y)
        test_y = le.transform(test_y)

        # initialise our trials dictionary
        trials = {}

        # loop over the number of trials to run
        for i in range(0, GeneralSettings().n_trials):
            # train the model
            logger.info('Training model {} of {}.'.format(i + 1, GeneralSettings().n_trials))
            model = RandomForestClassifier(GeneralSettings().n_estimators)
            model.fit(train_x, train_y)

            # make predictions on the testing data and initialise a dictionary to store our
            # computed metrics
            predictions = model.predict(test_x)
            metrics = {}

            # compute the confusion matrix and and use it to derive the raw accuracy, sensitivity,
            # and specificity
            cm = confusion_matrix(test_y, predictions).flatten()
            (tn, fp, fn, tp) = cm
            metrics["accuracy"] = (tp + tn) / float(cm.sum())
            metrics["sensitivity"] = tp / float(tp + fn)
            metrics["specificity"] = tn / float(tn + fp)

            # loop over the metrics
            for (k, v) in metrics.items():
                # update the trials dictionary with the list of values for the current metric
                l_values = trials.get(k, [])
                l_values.append(v)
                trials[k] = l_values

        # define a video capture object
        vid = cv2.VideoCapture(path + '/pics/cut_fast.mp4')
        last_frame = np.zeros((GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE), dtype=np.uint8)

        while vid.isOpened():

            # Capture the video frame by frame
            ret, frame = vid.read()

            if ret:

                # pre-process the image in the same manner we did earlier
                image = center_crop(frame, GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE)
                image = gray(image)
                image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # compute difference
                score, diff = ssim(image, last_frame, full=True)

                if score > float(0):

                    # quantify the image and make predictions based on the extracted features using
                    # the last trained Random Forest
                    features = quantify_image(image)
                    preds = model.predict([features])
                    label = le.inverse_transform(preds)[0]

                    # draw the colored class label on the output image and add it to the set of output images
                    color = (255, 0, 0) if label == 'on' else (0, 0, 255)

                    logger.info('Different! {} [{}]'.format(label, score))
                    last_frame = image

                    frame = center_crop(frame, GeneralSettings().IMG_SIZE, GeneralSettings().IMG_SIZE)
                    cv2.putText(frame,
                                str(label),
                                (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color,
                                2,
                                cv2.LINE_4)

                    # Display the resulting frame
                    cv2.imshow('Board', frame)
                    cv2.waitKey(1)

                else:
                    logger.info('Identical! [{}]'.format(score))

            else:
                break

        # After the loop release the cap object
        vid.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.exception(e)

    finally:
        tm.elapsed()
