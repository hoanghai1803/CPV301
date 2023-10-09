import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
from typing import List
import functools
from image_processing import ImageProcessing, Effect


def open_image() -> None:
    file_path = filedialog.askopenfilename()
    if file_path:
        image_states.clear()
        effect_list.clear()
        original_image = cv.imread(file_path)
        image_states.append(original_image)
        display_original_image()
        display_edited_image()


def process_image(effect: Effect, params=None) -> None:
    if params is None:
        params = []
    if not image_states:
        return

    # Get the current image t
    current_image = image_states[-1].copy()

    # Create an ImageProcessing instance
    ip = ImageProcessing(current_image)

    # Process the current image to create the new image
    new_image = np.zeros((1, 1, 1))

    if effect == Effect.WHITE_PATCH:
        new_image = ip.white_patch()
        white_patch_btn.config(state=tk.DISABLED)

    if effect == Effect.GRAY_WORLD:
        new_image = ip.gray_world()
        gray_world_btn.config(state=tk.DISABLED)

    if effect == Effect.GROUND_TRUTH:
        y, x = params[0:2]
        height, width = params[2:4]
        new_image = ip.ground_truth((y, x), (height, width))
        ground_truth_btn.config(state=tk.DISABLED)

    if effect == Effect.HISTOGRAM_EQUALIZATION:
        new_image = ip.histogram_equalization()
        histogram_equal_btn.config(state=tk.DISABLED)

    if effect == Effect.MEDIAN_FILTER:
        ksize = params[0]
        new_image = ip.median_filter(ksize)
        median_btn.config(state=tk.DISABLED)

    if effect == Effect.MEAN_FILTER:
        ksize = params[0]
        new_image = ip.mean_filter(ksize)
        mean_btn.config(state=tk.DISABLED)

    if effect == Effect.GAUSSIAN_SMOOTHING:
        ksize, sigma = params
        new_image = ip.gaussian_smoothing(3, 0.1)
        gaussian_btn.config(state=tk.DISABLED)

    if effect == Effect.HARRIS_CORNER:
        new_image = ip.harris_corner()
        harris_btn.config(state=tk.DISABLED)

    if effect == Effect.HOG:
        new_image = ip.hog()
        hog_btn.config(state=tk.DISABLED)

    if effect == Effect.CANNY:
        new_image = ip.canny()
        canny_btn.config(state=tk.DISABLED)

    if effect == Effect.HOUGH_TRANSFORM:
        new_image = ip.hough_transform()
        hough_btn.config(state=tk.DISABLED)

    image_states.append(new_image)
    effect_list.append(effect)

    display_edited_image()


def normalize_image(image: np.ndarray) -> ImageTk.PhotoImage:
    # Convert the OpenCV BGR image to RGB format
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Convert the image to a PIL Image
    pil_image = Image.fromarray(image_rgb)

    # Resize the image to fit within the maximum width while maintaining the aspect ratio
    pil_image.thumbnail((SCREEN_WIDTH // 2 - 150, SCREEN_WIDTH // 2))

    # Convert the PIL Image to a PhotoImage
    photo = ImageTk.PhotoImage(pil_image)

    return photo


def display_original_image() -> None:
    # Get the original image
    original_image = image_states[0].copy()

    # Get the Photo Image for exporting
    photo = normalize_image(original_image)

    original_image_label.config(image=photo)
    original_image_label.image = photo


def display_edited_image() -> None:
    # Get the last state of image
    edited_image = image_states[-1].copy()

    # Get the Photo Image for exporting
    photo = normalize_image(edited_image)

    edited_image_label.config(image=photo)
    edited_image_label.image = photo


def undo_action() -> None:
    if effect_list:
        last_effect = effect_list[-1]
        if last_effect == Effect.WHITE_PATCH:
            white_patch_btn.config(state=tk.NORMAL)

        if last_effect == Effect.GRAY_WORLD:
            gray_world_btn.config(state=tk.NORMAL)

        if last_effect == Effect.GROUND_TRUTH:
            ground_truth_btn.config(state=tk.NORMAL)

        if last_effect == Effect.HISTOGRAM_EQUALIZATION:
            histogram_equal_btn.config(state=tk.NORMAL)

        if last_effect == Effect.MEDIAN_FILTER:
            median_btn.config(state=tk.NORMAL)

        if last_effect == Effect.MEAN_FILTER:
            mean_btn.config(state=tk.NORMAL)

        if last_effect == Effect.GAUSSIAN_SMOOTHING:
            gaussian_btn.config(state=tk.NORMAL)

        if last_effect == Effect.HARRIS_CORNER:
            harris_btn.config(state=tk.NORMAL)

        if last_effect == Effect.HOG:
            hog_btn.config(state=tk.NORMAL)

        if last_effect == Effect.CANNY:
            canny_btn.config(state=tk.NORMAL)

        if last_effect == Effect.HOUGH_TRANSFORM:
            hough_btn.config(state=tk.NORMAL)

        image_states.pop()
        effect_list.pop()
        display_edited_image()


def handle_ground_truth():
    def apply_effect():
        try:
            y = int(y_entry.get())
            x = int(x_entry.get())
            height = int(height_entry.get())
            width = int(width_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
            return

        if y < 0 or x < 0 or width < 0 or height < 0 or y + height > image_height or x + width > image_width:
            messagebox.showerror("Invalid Input", "Invalid input. Try again!")
            return

        process_image(Effect.GROUND_TRUTH, [y, x, height, width])
        sub_window.destroy()

    sub_window = tk.Toplevel(root)
    sub_window.title('Ground Truth')
    sub_window.geometry("700x500")

    original_image = image_states[0].copy()
    image_height, image_width = original_image.shape[0], original_image.shape[1]
    shape_title = tk.Label(sub_window, text=f'The shape of image: {image_height}x{image_width}.', anchor='w')
    shape_title.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")

    top_left_label = tk.Label(
        sub_window,
        text=f"Enter the coordinates of the top-left of the patch [0:{image_height-1}, 0:{image_width-1}]:",
        anchor="w")
    top_left_label.grid(row=2, column=0, padx=(20, 0), pady=20, sticky="w")
    y_label = tk.Label(sub_window, text="Y:")
    y_label.grid(row=3, column=0, padx=(20, 0), pady=5, sticky="w")
    y_entry = tk.Entry(sub_window, bd=5, width=5)
    y_entry.grid(row=3, column=0, padx=(40, 0), pady=5, sticky="w")
    x_label = tk.Label(sub_window, text="X:")
    x_label.grid(row=3, column=0, padx=(120, 0), pady=5, sticky="w")
    x_entry = tk.Entry(sub_window, bd=5, width=5)
    x_entry.grid(row=3, column=0, padx=(140, 0), pady=5, sticky="w")

    shape_title = tk.Label(
        sub_window,
        text="Enter the shape of the patch:",
        anchor="w")
    shape_title.grid(row=4, column=0, padx=(20, 0), pady=20, sticky="w")
    height_label = tk.Label(sub_window, text="Height:")
    height_label.grid(row=5, column=0, padx=(20, 0), pady=5, sticky="w")
    height_entry = tk.Entry(sub_window, bd=5, width=5)
    height_entry.grid(row=5, column=0, padx=(80, 0), pady=5, sticky="w")
    width_label = tk.Label(sub_window, text="Width:")
    width_label.grid(row=5, column=0, padx=(160, 0), pady=5, sticky="w")
    width_entry = tk.Entry(sub_window, bd=5, width=5)
    width_entry.grid(row=5, column=0, padx=(220, 0), pady=5, sticky="w")

    # Create a button to apply the effect
    finish_button = tk.Button(sub_window, text='Apply',
                              command=apply_effect, cursor="hand2")
    finish_button.grid(row=6, column=0, padx=(40, 0), pady=20, sticky="w")


def handle_median_filter():
    def apply_effect():
        try:
            ksize = int(ksize_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
            return

        if ksize < 0 or ksize % 2 == 0:
            messagebox.showerror("Invalid Input", "Invalid input. Try again!")
            return

        process_image(Effect.MEDIAN_FILTER, [ksize])
        sub_window.destroy()

    sub_window = tk.Toplevel(root)
    sub_window.title('Median Filter')
    sub_window.geometry("700x500")

    main_title = tk.Label(
        sub_window,
        text="Enter size of the kernel (should be odd number):",
        anchor="w")
    main_title.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")
    ksize_label = tk.Label(sub_window, text="Size:")
    ksize_label.grid(row=1, column=0, padx=(20, 0), pady=5, sticky="w")
    ksize_entry = tk.Entry(sub_window, bd=5, width=5)
    ksize_entry.grid(row=1, column=0, padx=(60, 0), pady=5, sticky="w")

    # Create a button to apply the effect
    finish_button = tk.Button(sub_window, text='Apply',
                              command=apply_effect, cursor="hand2")
    finish_button.grid(row=2, column=0, padx=(40, 0), pady=20, sticky="w")


def handle_mean_filter():
    def apply_effect():
        try:
            ksize = int(ksize_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
            return

        if ksize < 0 or ksize % 2 == 0:
            messagebox.showerror("Invalid Input", "Invalid input. Try again!")
            return

        process_image(Effect.MEAN_FILTER, [ksize])
        sub_window.destroy()

    sub_window = tk.Toplevel(root)
    sub_window.title('Mean Filter')
    sub_window.geometry("700x500")

    main_title = tk.Label(
        sub_window,
        text="Enter size of the kernel (should be odd number):",
        anchor="w")
    main_title.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")
    ksize_label = tk.Label(sub_window, text="Size:")
    ksize_label.grid(row=1, column=0, padx=(20, 0), pady=5, sticky="w")
    ksize_entry = tk.Entry(sub_window, bd=5, width=5)
    ksize_entry.grid(row=1, column=0, padx=(60, 0), pady=5, sticky="w")

    # Create a button to apply the effect
    finish_button = tk.Button(sub_window, text='Apply',
                              command=apply_effect, cursor="hand2")
    finish_button.grid(row=2, column=0, padx=(40, 0), pady=20, sticky="w")


def handle_gaussian_smoothing():
    def apply_effect():
        try:
            ksize = int(ksize_entry.get())
            sigma = float(sigma_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
            return

        if ksize < 0 or ksize % 2 == 0 or sigma < 0:
            messagebox.showerror("Invalid Input", "Invalid input. Try again!")
            return

        process_image(Effect.GAUSSIAN_SMOOTHING, [ksize, sigma])
        sub_window.destroy()

    sub_window = tk.Toplevel(root)
    sub_window.title('Gaussian Smoothing')
    sub_window.geometry("700x500")

    main_title = tk.Label(
        sub_window,
        text="Enter the coefficients of Gaussian smoothing:",
        anchor="w")
    main_title.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")
    ksize_label = tk.Label(sub_window, text="Size:")
    ksize_label.grid(row=1, column=0, padx=(20, 0), pady=5, sticky="w")
    ksize_entry = tk.Entry(sub_window, bd=5, width=5)
    ksize_entry.grid(row=1, column=0, padx=(60, 0), pady=5, sticky="w")
    sigma_label = tk.Label(sub_window, text="Sigma:")
    sigma_label.grid(row=1, column=0, padx=(120, 0), pady=5, sticky="w")
    sigma_entry = tk.Entry(sub_window, bd=5, width=5)
    sigma_entry.grid(row=1, column=0, padx=(180, 0), pady=5, sticky="w")

    # Create a button to apply the effect
    finish_button = tk.Button(sub_window, text='Apply',
                              command=apply_effect, cursor="hand2")
    finish_button.grid(row=2, column=0, padx=(40, 0), pady=20, sticky="w")


root = tk.Tk()
root.title("Image Processing App")

# CONSTANTS
SCREEN_HEIGHT = root.winfo_screenheight()   # Height of the screen
SCREEN_WIDTH = root.winfo_screenwidth()     # Width of the screen

# Global variables
image_states: List[np.ndarray] = []         # List of image states for rolling back
effect_list: List[Effect] = []              # List of effects for rolling back

# Set fullscreen window
root.geometry(f'{SCREEN_WIDTH}x{SCREEN_HEIGHT}')

# The original image is displayed on the left
original_frame = tk.Frame(root)
original_frame.pack(side=tk.LEFT)

# The edited image is displayed on the right
edited_frame = tk.Frame(root)
edited_frame.pack(side=tk.RIGHT)

original_image_label = tk.Label(original_frame)
original_image_label.pack()

edited_image_label = tk.Label(edited_frame)
edited_image_label.pack()

open_btn = tk.Button(root, text="Open new image", command=open_image, cursor='hand2')
open_btn.pack(pady=20)

white_patch_btn = tk.Button(root, text="White Patch", command=functools.partial(process_image, Effect.WHITE_PATCH),
                            cursor='hand2')
white_patch_btn.pack(pady=20)

gray_world_btn = tk.Button(root, text="Gray World", command=functools.partial(process_image, Effect.GRAY_WORLD),
                           cursor='hand2')
gray_world_btn.pack(pady=20)

ground_truth_btn = tk.Button(root, text="Ground Truth", command=handle_ground_truth, cursor='hand2')
ground_truth_btn.pack(pady=20)

histogram_equal_btn = tk.Button(root, text="Histogram Equalization",
                                command=functools.partial(process_image, Effect.HISTOGRAM_EQUALIZATION), cursor='hand2')
histogram_equal_btn.pack(pady=20)

median_btn = tk.Button(root, text="Median Filter", command=handle_median_filter, cursor='hand2')
median_btn.pack(pady=20)

mean_btn = tk.Button(root, text="Mean Filter", command=handle_mean_filter, cursor='hand2')
mean_btn.pack(pady=20)

gaussian_btn = tk.Button(root, text="Gaussian Smoothing", command=handle_gaussian_smoothing, cursor='hand2')
gaussian_btn.pack(pady=20)

harris_btn = tk.Button(root, text="Harris Corner", command=functools.partial(process_image, Effect.HARRIS_CORNER),
                            cursor='hand2')
harris_btn.pack(pady=20)

hog_btn = tk.Button(root, text="HOG", command=functools.partial(process_image, Effect.HOG),
                            cursor='hand2')
hog_btn.pack(pady=20)

canny_btn = tk.Button(root, text="Canny Operator", command=functools.partial(process_image, Effect.CANNY),
                            cursor='hand2')
canny_btn.pack(pady=20)

hough_btn = tk.Button(root, text="Hough Transform", command=functools.partial(process_image, Effect.HOUGH_TRANSFORM),
                            cursor='hand2')
hough_btn.pack(pady=20)


undo_btn = tk.Button(root, text="Undo", command=undo_action, cursor='hand2')
undo_btn.pack(pady=20)

root.mainloop()
