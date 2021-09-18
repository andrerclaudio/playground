# Build-in modules
import logging
from datetime import timedelta

# Added modules
import cv2
import numpy as np
from pytictoc import TicToc

# Project modules


# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)

IMAGE_PATH = "./pics/lights_2.png"


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
    """

    """

    width = int(image.shape[1] * proportion / 100)
    height = int(image.shape[0] * proportion / 100)
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    logger.debug(f'Resized dimensions: {resized.shape}')
    return resized


def centroid_histogram(clt):
    """

    """
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def application():
    """" All application has its initialization from here """
    logger.info('Main application is running!')

    tm = ElapsedTime()
    image = False

    try:

        # define a video capture object
        vid = cv2.VideoCapture(0)

        while True:

            # Capture the video frame by frame
            ret, frame = vid.read()

            # Display the resulting frame
            # cv2.imshow('frame', frame)

            # Convert to RGB format
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Choose the values based on the color on the point/mark
            lower_green = np.array([0, 150, 0])
            upper_green = np.array([[140, 255, 35]])
            filter_green = cv2.inRange(img_rgb, lower_green, upper_green)

            # cv2.imshow("Lights", filter_green)
            # cv2.waitKey(0)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(filter_green, cv2.MORPH_OPEN, kernel, iterations=1)

            cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            for c in cnts:
                # compute the center of the contour
                m = cv2.moments(c)
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])

                # draw the center of the shape on the image
                cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 2)
                # cv2.drawContours(output, [c], 0, (255, 255, 255), 1)

            cv2.imshow("Lights", frame)

            # the 'q' button is set as the quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # image = cv2.imread(IMAGE_PATH)
        # # image = resize(image, proportion=40)
        # output = image.copy()
        # logger.info(f'Original dimensions: {image.shape}')
        #
        # # Convert to RGB format
        # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # # Choose the values based on the color on the point/mark
        # lower_green = np.array([0, 145, 0])
        # upper_green = np.array([210, 255, 75])
        # filter_green = cv2.inRange(img_rgb, lower_green, upper_green)
        # # cv2.imshow("Filter", filter_green)
        #
        # # Bitwise-AND mask and original image
        # # masked_green = cv2.bitwise_and(img_rgb, img_rgb, mask=filter_green)
        # # cv2.imshow("Mask", masked_green)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # opening = cv2.morphologyEx(filter_green, cv2.MORPH_OPEN, kernel, iterations=1)
        #
        # cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #
        # for c in cnts:
        #     # compute the center of the contour
        #     m = cv2.moments(c)
        #     cx = int(m["m10"] / m["m00"])
        #     cy = int(m["m01"] / m["m00"])
        #
        #     # draw the center of the shape on the image
        #     cv2.circle(output, (cx, cy), 15, (0, 0, 255), 2)
        #     # cv2.drawContours(output, [c], 0, (255, 255, 255), 1)
        #
        # cv2.imshow("Output", output)
        # cv2.waitKey(0)

        # After the loop release the cap object
        vid.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.exception(e, exc_info=False)

    finally:
        tm.elapsed()
        return image
