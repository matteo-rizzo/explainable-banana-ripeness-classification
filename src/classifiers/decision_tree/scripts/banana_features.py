import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.stats import multivariate_normal
from skimage.color import rgb2yuv


def compute_local_contrast(channel: np.ndarray, window_size: int = 3) -> np.ndarray:
    assert window_size % 2 != 0, "window_size must be odd"

    # Convert the image to YUV and extract the Y (luma/brightness) channel
    # yuv_image = rgb2yuv(image)
    # y_channel = yuv_image[..., 0]

    channel = np.nan_to_num(channel)
    # Compute the maximum and minimum intensities in each patch
    i_max = maximum_filter(channel, size=window_size)
    i_min = minimum_filter(channel, size=window_size)

    # Compute the contrast in each patch
    contrast = (i_max - i_min) / (i_max + i_min + 1e-6)  # small constant to prevent division by zero

    return contrast


def visualize_mask(original_image: np.ndarray, mask: np.ndarray):
    """
    Visualize the original image and the masked image side by side

    :param original_image: 3D numpy array with the original RGB image data
    :param mask: 2D numpy array with the binary mask
    """
    # Create a copy of the original image
    masked_image = original_image.copy()

    # Apply the mask to the image (this will set all non-brown pixels to black)
    for i in range(3):  # Loop over each color channel
        masked_image[..., i] *= mask

    # Create a figure with two subplots
    _, ax = plt.subplots(1, 2)

    # Display the original image in the first subplot
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display the masked image in the second subplot
    ax[1].imshow(masked_image)
    ax[1].set_title('Masked Image')
    ax[1].axis('off')

    # Show the figure
    plt.show()


def make_average(x: np.ndarray) -> np.ndarray:
    # Put channels to nan where they are all masked
    x[~x.any(axis=-1)] = np.nan
    # Take mean, resulting in mean over 3 channels for each image
    avg_color = np.nanmean(x, axis=(0, 1))
    avg_color[np.isnan(avg_color)] = .0
    return avg_color


def brownness(x: np.ndarray, data: dict[str, list[float]], strategy: str):
    """
    Compute features of brown color

    :param x: 3D numpy array with RGB image data in [0, 1]
    :param data: dictionaries where to place the channel features
    :param strategy: whether to use brown shades or contrast
    :return: the dictionary with updated features
    """
    ox = x
    x = rgb2yuv(x, channel_axis=-1)
    target_u = -.1
    target_v = .1
    u = 1
    v = 2

    if strategy == "brown":
        # Define mean and covariance matrix for the Gaussian distribution
        mean_color = np.array([target_u, target_v])
        covariance_matrix = np.array([[0.05, 0], [0, 0.05]])

        # Create a multivariate normal distribution
        rv = multivariate_normal(mean_color, covariance_matrix)

        # Compute the probability density function for each pixel
        pdf = rv.pdf(x[..., u:(v + 1)])

        # Get the indices of the pixels that are within a certain range (e.g., within one standard deviation)
        brown_pixels = np.where(pdf > rv.pdf(mean_color - 0.1 * np.sqrt(np.diag(covariance_matrix))))

        # Create a binary mask with the same shape as the input image
        brown_mask = np.zeros_like(x[..., 0], dtype=bool)
        # Set the mask to True at the positions of the brown pixels
        brown_mask[brown_pixels] = True

        brown_avg = make_average(x[..., u:(v + 1)])
        data["bu"].append(brown_avg[0])
        data["bv"].append(brown_avg[1])
    else:
        brown_mask = compute_local_contrast(x[..., 0])

    brown_num = brown_mask.sum() / (x.shape[0] * x.shape[1])
    data["bn"].append(brown_num)

    if brown_num > .3:
        visualize_mask(ox, brown_mask)

    return data


def rgb_mean(x: np.ndarray, data: dict[str, list[float]]):
    """
    Average color in RGB space

    :param x: 3D numpy array with RGB image data in [0, 1]
    :param data: dictionaries where to place the channel features
    :return: the dictionary with updated features
    """
    avg_color = make_average(x)

    data["r"].append(avg_color[0])
    data["g"].append(avg_color[1])
    data["b"].append(avg_color[2])

    return data


def uv_mean(x: np.ndarray, data: dict[str, list[float]]):
    """
    Average color in YUV

    :param x: 3D numpy array with RGB image data in [0, 1]
    :param data: dictionaries where to place the channel features
    :return: the dictionary with updated features
    """
    x = rgb2yuv(x, channel_axis=-1)

    avg_color = make_average(x)

    # data["l"].append(avg_color[0])
    data["u"].append(avg_color[1])
    data["v"].append(avg_color[2])

    return data
