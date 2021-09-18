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

IMAGE_PATH = "./pics/lights.png"


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


def application():
    """" All application has its initialization from here """
    logger.info('Main application is running!')

    tm = ElapsedTime()
    image = False

    try:

        image = cv2.imread(IMAGE_PATH)
        output = image.copy()
        logger.info(f'Original dimensions: {image.shape}')

        # Convert to RGB format
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Choose the values based on the color on the point/mark
        lower_green = np.array([0, 200, 0])
        upper_green = np.array([255, 255, 255])
        mask = cv2.inRange(img_rgb, lower_green, upper_green)

        # Bitwise-AND mask and original image
        masked_green = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(masked_green, cv2.COLOR_BGR2GRAY)

        # detect circles in the image
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            # show the output image
            cv2.imshow("output", np.hstack([image, output]))

        cv2.imshow("Lights", masked_green)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    except Exception as e:
        logger.exception(e, exc_info=False)

    finally:
        tm.elapsed()
        return image
