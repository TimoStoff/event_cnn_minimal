from .timers import Timer, CudaTimer
import numpy as np
import cv2


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = np.expand_dims(X[dy, :], axis=0)
    elif dy < 0:
        X[dy:, :] = np.expand_dims(X[dy, :], axis=0)
    if dx > 0:
        X[:, :dx] = np.expand_dims(X[:, dx], axis=1)
    elif dx < 0:
        X[:, dx:] = np.expand_dims(X[:, dx], axis=1)
    return X


def upsample_color_image(grayscale_highres, color_lowres_bgr, colorspace='LAB'):
    """
    Generate a high res color image from a high res grayscale image, and a low res color image,
    using the trick described in:
    http://www.planetary.org/blogs/emily-lakdawalla/2013/04231204-image-processing-colorizing-images.html
    """
    assert(len(grayscale_highres.shape) == 2)
    assert(len(color_lowres_bgr.shape) == 3 and color_lowres_bgr.shape[2] == 3)

    if colorspace == 'LAB':
        # convert color image to LAB space
        lab = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2LAB)
        # replace lightness channel with the highres image
        lab[:, :, 0] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=lab, code=cv2.COLOR_LAB2BGR)
    elif colorspace == 'HSV':
        # convert color image to HSV space
        hsv = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HSV)
        # replace value channel with the highres image
        hsv[:, :, 2] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hsv, code=cv2.COLOR_HSV2BGR)
    elif colorspace == 'HLS':
        # convert color image to HLS space
        hls = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HLS)
        # replace lightness channel with the highres image
        hls[:, :, 1] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hls, code=cv2.COLOR_HLS2BGR)

    return color_highres_bgr


def merge_channels_into_color_image(channels):
    """
    Combine a full resolution grayscale reconstruction and four color channels at half resolution
    into a color image at full resolution.

    :param channels: dictionary containing the four color reconstructions (at quarter resolution),
                     and the full resolution grayscale reconstruction.
    :return a color image at full resolution
    """

    with Timer('Merge color channels'):

        assert('R' in channels)
        assert('G' in channels)
        assert('W' in channels)
        assert('B' in channels)
        assert('grayscale' in channels)

        # upsample each channel independently
        for channel in ['R', 'G', 'W', 'B']:
            channels[channel] = cv2.resize(channels[channel], dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Shift the channels so that they all have the same origin
        channels['B'] = shift_image(channels['B'], dx=1, dy=1)
        channels['G'] = shift_image(channels['G'], dx=1, dy=0)
        channels['W'] = shift_image(channels['W'], dx=0, dy=1)

        # reconstruct the color image at half the resolution using the reconstructed channels RGBW
        reconstruction_bgr = np.dstack([channels['B'],
                                        cv2.addWeighted(src1=channels['G'], alpha=0.5,
                                                        src2=channels['W'], beta=0.5,
                                                        gamma=0.0, dtype=cv2.CV_8U),
                                        channels['R']])

        reconstruction_grayscale = channels['grayscale']

        # combine the full res grayscale resolution with the low res to get a full res color image
        upsampled_img = upsample_color_image(reconstruction_grayscale, reconstruction_bgr)
    return upsampled_img
