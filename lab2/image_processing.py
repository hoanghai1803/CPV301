import numpy as np
from typing import Tuple, Union
from enum import Enum


class ImageProcessing:
    def __init__(self, image: np.array) -> None:
        self.__image = image.astype(np.float32)
        self.__height = image.shape[0]
        self.__width = image.shape[1]
        self.__channels = image.shape[2]

    @property
    def image(self) -> int:
        return self.__image
    
    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def channels(self) -> int:
        return self.__channels

    @image.setter
    def image(self, image):
        self.__image = image
        self.__height = image.shape[0]
        self.__width = image.shape[1]
        self.__channels = image.shape[2]

    def white_patch(self) -> np.ndarray:
        # Get the max of R, G, B in entire image
        b_max = np.max(self.__image[:, :, 0])
        g_max = np.max(self.__image[:, :, 1])
        r_max = np.max(self.__image[:, :, 2])

        # Apply the scaling factors to the entire image
        balanced_image = self.__image.copy()
        balanced_image[:, :, 0] /= b_max
        balanced_image[:, :, 1] /= g_max
        balanced_image[:, :, 2] /= r_max

        # Ensure that pixel values are in the valid range [0, 1]
        balanced_image = np.clip(balanced_image, 0, 1)

        # Convert the image back to uint8 data type
        balanced_image = (balanced_image * 255).astype(np.uint8)

        return balanced_image

    def gray_world(self) -> np.ndarray:
        # Calculate mean of the whole image
        image_mean = np.mean(self.__image, dtype=np.float32)
        b_mean = np.mean(self.__image[:, :, 0], dtype=np.float32)
        g_mean = np.mean(self.__image[:, :, 1], dtype=np.float32)
        r_mean = np.mean(self.__image[:, :, 2], dtype=np.float32)

        # Balance each pixel of the image by the mean
        balanced_image = self.__image.copy()
        balanced_image[:, :, 0] *= image_mean / b_mean
        balanced_image[:, :, 1] *= image_mean / g_mean
        balanced_image[:, :, 2] *= image_mean / r_mean

        # Ensure that pixel values are in the valid range [0, 255]
        balanced_image = np.clip(balanced_image, 0, 255)

        # Convert the image back to uint8 data type
        balanced_image = balanced_image.astype(np.uint8)

        return balanced_image

    def ground_truth(self, patch_coordinates: tuple[int, int], patch_size: tuple[int, int]) -> np.ndarray:
        # Get patch coordinates
        y, x = patch_coordinates

        # Get patch size
        patch_height, patch_width = patch_size

        # Ensure that the patch is inside the image
        assert y >= 0 and y + patch_height <= self.__height
        assert x >= 0 and x + patch_width <= self.__width

        # Extract the patch
        patch = self.__image[y:y+patch_height-1, x:x+patch_width-1]

        # Calculate mean of the patch
        patch_mean = np.mean(patch, dtype=np.float32)
        b_patch_mean = np.mean(patch[:, :, 0], dtype=np.float32)
        g_patch_mean = np.mean(patch[:, :, 1], dtype=np.float32)
        r_patch_mean = np.mean(patch[:, :, 2], dtype=np.float32)

        # Balance each pixel of the image by the patch mean
        balanced_image = self.__image.copy()
        balanced_image[:, :, 0] *= patch_mean / b_patch_mean
        balanced_image[:, :, 1] *= patch_mean / g_patch_mean
        balanced_image[:, :, 2] *= patch_mean / r_patch_mean

        # Ensure that pixel values are in the valid range [0, 255]
        balanced_image = np.clip(balanced_image, 0, 255)

        # Convert the image back to uint8 data type
        balanced_image = balanced_image.astype(np.uint8)

        return balanced_image

    def histogram_equalization(self) -> np.ndarray:
        # Calculate histogram and CDF of the image
        hist, bins = np.histogram(self.__image.flatten(), 256, (0, 256))
        cdf = hist.cumsum()

        # Mask array concept array from Numpy. For masked array, all operations are performed on non-masked elements.
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')

        return cdf[self.__image.astype('uint8')]

    def __zero_pad(self, kernel_shape: Union[Tuple[int, ...]]) -> np.ndarray:
        # Get the dimension of the kernel
        kernel_width, kernel_height = kernel_shape

        # Get padding size
        left_pad, right_pad = (kernel_width - 1) // 2, (kernel_width - 1) // 2 + (kernel_width - 1) % 2
        top_pad, bottom_pad = (kernel_height - 1) // 2, (kernel_height - 1) // 2 + (kernel_height - 1) % 2
        
        # Create an empty array with the new dimensions after padding
        padded_image = np.zeros((self.__height+kernel_height-1, self.__width+kernel_width-1,
                                 self.__channels), dtype=self.__image.dtype)

        # Copy the original image into the padded region
        padded_image[top_pad:top_pad+self.__height, left_pad:left_pad+self.__width, :] = self.__image
        
        return padded_image

    def median_filter(self, ksize: int) -> np.ndarray:
        # Ensure that kernel_size is an odd number
        assert ksize % 2 == 1

        # Get the 0-pad image to prevent from reducing dimension of the output image
        padded_image = self.__zero_pad((ksize, ksize))

        # Calculate the output size
        output_height, output_width, output_channels = self.__image.shape

        # Create an output image of the same size as the input
        output = np.zeros_like(self.__image)

        # Apply the mean filter to the image
        for channel in range(output_channels):
            for i in range(output_height):
                for j in range(output_width):
                    # Extract a (ksize x ksize) neighborhood around the pixel
                    neighborhood = padded_image[i:i+ksize, j:j+ksize, channel]

                    # Apply median filter by getting the median of the neighborhood
                    output[i, j, channel] = np.median(neighborhood)

        return output.astype('uint8')

    def convolution(self, kernel: np.ndarray) -> np.ndarray:
        # Get the 0-pad image to prevent from reducing dimensions of the output image
        padded_image = self.__zero_pad(kernel.shape)

        # Get shape of the kernel
        kernel_height, kernel_width = kernel.shape

        # Calculate the output size
        output_height, output_width, output_channels = self.__image.shape

        # Initialize the result matrix
        output = np.zeros_like(self.__image)

        # Perform the convolution
        for channel in range(output_channels):
            for i in range(output_height):
                for j in range(output_width):
                    # Extract a patch from the image
                    image_patch = padded_image[i:i+kernel_height, j:j+kernel_width, channel]

                    # Perform element-wise multiplication and sum
                    output[i, j, channel] = np.sum(image_patch * kernel)

        return output.astype('uint8')

    def mean_filter(self, ksize: int) -> np.ndarray:
        # Create a kernel for mean filter
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize*ksize)

        # Invoke the `convolution` method to do mean filter
        return self.convolution(kernel)

    def get_gaussian_kernel(self, ksize: int, sigma: float) -> np.ndarray:
        assert ksize % 2 == 1

        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = (ksize // 2, ksize // 2)
        for i in range(ksize):
            for j in range(ksize):
                x, y = center[0] - i, center[1] - j
                kernel[i, j] = 1.0 / (2 * np.pi * sigma * sigma) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))

        return kernel

    def gaussian_smoothing(self, ksize: int, sigma: float) -> np.ndarray:
        assert ksize % 2 == 1

        # Get the Gaussian smoothing kernel
        kernel = self.get_gaussian_kernel(ksize, sigma)

        # Invoke the `convolution` method to do Gaussian smoothing
        return self.convolution(kernel)


class Effect(Enum):
    WHITE_PATCH = 1
    GRAY_WORLD = 2
    GROUND_TRUTH = 3
    HISTOGRAM_EQUALIZATION = 4
    MEDIAN_FILTER = 5
    MEAN_FILTER = 6
    GAUSSIAN_SMOOTHING = 7
