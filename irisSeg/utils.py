import numpy as np
from scipy.signal import convolve

_round = lambda x: np.round(x).astype(np.uint8)


def search(image, rmin, rmax, x, y, feature):
    '''
        function to detect the pupil  boundary
        it searches a certain subset of the image
        with a given radius range(rmin,rmax)
        around a 10*10 neighbourhood of the point x,y given as input

    :param image: image to be processed
    :param rmin: min radius
    :param rmax: max radius
    :param x: x - coord of center point
    :param y: y - coord of center point
    :param feature: 'pupil' or 'iris'
    :return: Center coord followed by radius
    '''
    sigma = 0.5
    maxRadius = np.zeros(image.shape)
    maxBlur = np.zeros(image.shape)
    for i in np.arange(int(x) - 5, int(x) + 5):
        for j in np.arange(int(y) - 5, int(y) + 5):
            max_blur, max_blur_radius, blur = partialDerivative(image, [i, j], rmin, rmax, sigma, 600, feature)
            maxRadius[i, j] = max_blur_radius
            maxBlur[i, j] = max_blur
    X, Y = np.where(maxBlur == maxBlur.max())
    radius = maxRadius[X, Y]
    coordPupil = np.array([X, Y, radius])
    return coordPupil


def NormalLineIntegral(image, coord, r, n, feature):
    """
    Calculate the normalized line integral around a circular contour.
    Handles out-of-bound indices and ensures robust computation.

    Parameters:
        image (ndarray): Input image.
        coord (list): Center coordinates [x, y].
        r (int): Radius of the circle.
        n (int): Number of sides of the polygon.
        feature (str): Feature ('pupil' or 'iris').

    Returns:
        float: Normalized line integral value.
    """
    # Calculate the angle subtended by each polygon segment
    theta = (2 * np.pi) / n
    rows, cols = image.shape

    # Generate angles and calculate x, y coordinates of the circle's boundary
    angle = np.arange(theta, 2 * np.pi, theta)
    x = coord[0] - r * np.sin(angle)
    y = coord[1] + r * np.cos(angle)

    # Ensure coordinates are arrays for consistency
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    y = np.array(y) if not isinstance(y, np.ndarray) else y

    # Validate and clip indices to stay within image bounds
    x = np.clip(x, 0, rows - 1).astype(np.uint16)
    y = np.clip(y, 0, cols - 1).astype(np.uint16)

    if feature == 'pupil':
        # Sum pixel values around the entire circle
        s = 0
        for i in range(len(x)):
            s += image[x[i], y[i]]
        line = s / n  # Normalize by the circumference
        return line

    elif feature == 'iris':
        # Sum pixel values over specific arc segments for iris
        s = 0
        # Top segment
        for i in range(1, _round(n / 8)):
            s += image[x[i], y[i]]
        # Middle lateral segments
        for i in range(_round(3 * n / 8) + 1, _round(5 * n / 8)):
            s += image[x[i], y[i]]
        # Bottom segment
        for i in range(_round(7 * n / 8) + 1, len(x)):
            s += image[x[i], y[i]]
        line = (2 * s) / n  # Normalize
        return line

    else:
        raise ValueError("Invalid feature type. Use 'pupil' or 'iris'.")




def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def partialDerivative(image, coord, rmin, rmax, sigma, n, feature):
    '''
        calculates the partial derivative of the normailzed line integral
        holding the centre coordinates constant
        and then smooths it by a gaussian of appropriate sigma
        rmin and rmax are the minimum and maximum values of radii expected
        function also returns the maximum value of blur and the corresponding radius
        with finite differnce vector blur

    :param image: preprocessed image
    :param coord: centre coordinates
    :param rmin: min radius
    :param rmax: max radius
    :param sigma: standard deviation of the gaussian
    :param n: number of sides of the polygon(for LineIntegral)
    :param feature: pupil or Iris
    :return: It gives finite differences vector, max value of blur
            and radius at max blur
    '''
    R = np.arange(rmin, rmax)
    count = R.shape[0]

    lineIntegral = []
    # lineIntegral = np.empty(7)

    for k in np.arange(0, count):
        # computing the normalized line integral for each radius
        temp = NormalLineIntegral(image, coord, R[k], n, feature)
        if temp == 0:
            # this case occurs iff the radius takes the circle out of the image
            # In this case,L is deleted as shown below and no more radii are taken for computation
            # (for that particular centre point).This is accomplished using the break statement
            break
        else:
            lineIntegral.append(temp)
            # np.append(lineIntegral,temp)
    if not isinstance(lineIntegral, np.ndarray):
        lineIntegral = np.array(lineIntegral)

    disc_diff = np.diff(lineIntegral)
    D = np.concatenate(([0], disc_diff))  # append one element at the beginning

    if sigma == 'inf':
        kernel = np.ones(7) / 7
    else:
        kernel = matlab_style_gauss2D([1, 5], sigma)  # generates a 5 member 1-D gaussian
        kernel = np.reshape(kernel, kernel.shape[1], order='F')

    blur = np.abs(convolve(D, kernel, 'same'))
    # Smooths the D vecor by 1-D convolution

    values, index = blur.max(0), blur.argmax(0)
    max_blur_radius = R[index]
    max_blur = blur[index]
    return max_blur, max_blur_radius, blur


def drawcircle(I, C, r, n=600):
    '''
        generate the pixels on the boundary of a regular polygon of n sides
        the polygon approximates a circle of radius r and is used to draw the circle
    :param I: image to be processed
    :param C: [x,y] Centre coordinates of the circumcircle
    :param r: radius of the circumcircle
    :param n: no of sides
    :return: Image with circle
    '''
    theta = (2 * np.pi) / n
    rows, cols = I.shape
    angle = np.arange(theta, 2 * np.pi, theta)

    x = C[0] - r * np.sin(angle)
    y = C[1] + r * np.cos(angle)

    if np.any(x >= rows) or np.any(y >= cols) or np.any(x <= 1) or np.any(y <= 1):
        return I
    for i in np.arange(1, n - 1):
        I[np.round(x[i]).astype(np.uint8), np.round(y[i]).astype(np.uint8)] = 1
    return I
