# Build-in modules
import logging
from datetime import timedelta

# Added modules
import cv2
from pytictoc import TicToc

# Project modules


# Print in software terminal
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(process)d | %(name)s | %(levelname)s:  %(message)s',
                    datefmt='%d/%b/%Y - %H:%M:%S')

logger = logging.getLogger(__name__)

IMAGE_PATH = "./pics/image.jpg"
CASCADE_PATH = "haarcascade_frontalface_default.xml"


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

    # Create the haar cascade
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    image = False

    try:

        image = cv2.imread(IMAGE_PATH)
        logger.info(f'Original dimensions: {image.shape}')

        scale_percent = 20  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image and show
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        logger.debug(f'Resized dimensions: {resized.shape}')

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        logger.info('Found {0} faces!'.format(len(faces)))

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Faces found", resized)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    except Exception as e:
        logger.exception(e, exc_info=False)

    finally:
        tm.elapsed()
        return image
