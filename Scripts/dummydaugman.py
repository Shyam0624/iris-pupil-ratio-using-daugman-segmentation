import numpy as np
import cv2
from skimage.morphology import erosion
import matplotlib.pyplot as plt
from irisSeg.utils import partialDerivative, search, drawcircle
from PIL import Image

try:
    from PIL import Resampling  # Newer Pillow versions
    ANTIALIAS = Resampling.LANCZOS
except ImportError:
    try:
        ANTIALIAS = Image.ANTIALIAS  # Older Pillow versions
    except AttributeError:
        raise ImportError("Your Pillow version is incompatible. Please install Pillow 9.5.0 or upgrade to the latest version.")
# Replace this line:
# from scipy.misc import imresize


def imresize(img, scale):
    """
    Resize the image using PIL's resize method.
    :param img: Input image (NumPy array or PIL.Image object)
    :param scale: Scaling factor (float)
    :return: Resized image (PIL.Image object)
    """
    # Ensure img is a PIL Image object
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Calculate the new size
    size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(size, ANTIALIAS)
    return img_resized


# this function returns the indexes that satisfy certain function func
indices = lambda a, func: [i for (i, val) in enumerate(a) if func(val)]


#  This function replicates the matlab function rgb2gray
rgb2gray = lambda x: np.dot(x[..., :3], [0.2989, 0.5870, 0.1140])


#  This function replicates the matlab function im2double
im2double = lambda im: im.astype(float) / np.iinfo(im.dtype).max  # Divide all values by the largest possible value in the datatype

def irisSeg(image_or_path, rmin, rmax, view_output=False):
    """
    Segments the iris and pupil from the input image using Daugman's method.

    :param image_or_path: Image file path or NumPy array
    :param rmin: Minimum radius of the iris
    :param rmax: Maximum radius of the iris
    :param view_output: If True, displays the segmented image
    :return: Coordinates of the iris and pupil, and the segmented image
    """
    # Load the image if a file path is provided
    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_or_path}")
    elif isinstance(image_or_path, np.ndarray):
        if len(image_or_path.shape) == 3:  # Convert to grayscale if RGB
            image = cv2.cvtColor(image_or_path, cv2.COLOR_BGR2GRAY)
        else:
            image = image_or_path
    else:
        raise ValueError("Invalid input: Provide a file path or NumPy array.")

    # Convert to float [0, 1]
    image = im2double(image)

    # Convert to binary for erosion
    binary_image = (image < 0.5).astype(np.uint8)

    # Apply morphological erosion
    eroded_image = erosion(binary_image)

    rows, cols = image.shape

    # Identify potential center coordinates
    X, Y = np.where(image < 0.5)  # Extract low-intensity regions

    nan = -99999
    valid_indices = []
    for k in range(len(X)):
        if rmin <= X[k] <= rows - rmin and rmin <= Y[k] <= cols - rmin:
            A = image[max(0, X[k] - 1):X[k] + 2, max(0, Y[k] - 1):Y[k] + 2]
            if image[X[k], Y[k]] == A.min():
                valid_indices.append(k)

    X, Y = X[valid_indices], Y[valid_indices]

    # Initialize blur metrics
    maxBlur = np.zeros((rows, cols))
    maxRadius = np.zeros((rows, cols))

    for j in range(len(X)):
        max_blur, max_blur_radius, _ = partialDerivative(image, [X[j], Y[j]], rmin, rmax, 'inf', 600, 'iris')
        maxBlur[X[j], Y[j]] = max_blur
        maxRadius[X[j], Y[j]] = max_blur_radius

    # Locate the iris center
    x, y = np.unravel_index(np.argmax(maxBlur), maxBlur.shape)

    # Define scale before using it
    scale = 1.0  # Default scale value (1.0 means no scaling)

    coord_iris = search(image, rmin, rmax, x, y, 'iris') / scale
    coord_pupil = search(
        image,
        int(0.1 * maxRadius[x, y]),
        int(0.8 * maxRadius[x, y]),
        coord_iris[0] * scale,
        coord_iris[1] * scale,
        'pupil',
    ) / scale

    # Ensure pimage is defined (it was missing in the original code)
    pimage = np.copy(image)

    # Draw the detected circles
    segmented_img = drawcircle(pimage, [coord_iris[0], coord_iris[1]], coord_iris[2], 600)
    segmented_img = drawcircle(segmented_img, [coord_pupil[0], coord_pupil[1]], coord_pupil[2], 600)

    # Optionally display the output
    if view_output:
        plt.imshow(segmented_img, cmap='gray')
        plt.title("Segmented Iris and Pupil")
        plt.axis("off")
        plt.show()

    return coord_iris, coord_pupil, segmented_img


if __name__ == '__main__':
    # Example usage of the irisSeg function
    coord_iris, coord_pupil, output_image = irisSeg('Data/sample_img.jpg', 40, 70, view_output=True)

    print("Iris Coordinates:", coord_iris)
    print("Pupil Coordinates:", coord_pupil)
