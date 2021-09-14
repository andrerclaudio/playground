# Build-in modules
import logging
from datetime import timedelta

import cv2
# Added modules
from pytictoc import TicToc

# Project modules


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


def application():
    """" All application has its initialization from here """
    logger.info('Main application is running!')

    tm = ElapsedTime()
    image = False

    try:
        filename = "./pics/image.jpg"
        image = cv2.imread(filename)
        logger.info(f'Original dimensions: {image.shape}')

        scale_percent = 20  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image and show
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Resized image", resized)
        logger.info(f'Resized dimensions: {resized.shape}')

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        logger.exception(e, exc_info=False)

    finally:
        tm.elapsed()
        return image
