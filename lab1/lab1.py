import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coordinates(self):
        return [self.x, self.y]

    def to_homogeneous(self) -> np.ndarray:
        return np.array([self.x, self.y, 1.0], dtype=np.float32)

    def translate_coordinate(self):
        return Point(self.x-380, -(self.y-380))

    def actual_coordinate(self):
        return Point(self.x+380, -self.y+380)
    

    def transform(self, mat: np.ndarray):
        transformed_point = np.dot(mat, self.to_homogeneous())
        u, v, w = transformed_point
        return Point(u/w, v/w)

    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f})"


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Quadrangle:
    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point):
        points = [p1, p2, p3, p4]
        # Sort points based on their y-coordinate
        sorted_points = sorted(points, key=lambda p: p.y)
        # Split the sorted points into two rows
        bottom_row = sorted_points[:2]
        top_row = sorted_points[2:]
        # Sort each row by x-coordinate to get the final order
        bottom_left, bottom_right = sorted(bottom_row, key=lambda p: p.x)
        top_left, top_right = sorted(top_row, key=lambda p: p.x)
        self.p1, self.p2, self.p3, self.p4 = top_left, top_right, bottom_right, bottom_left

    def transform(self, matrix):
        p1 = self.p1.transform(matrix)
        p2 = self.p2.transform(matrix)
        p3 = self.p3.transform(matrix)
        p4 = self.p4.transform(matrix)
        return Quadrangle(p1, p2, p3, p4)
    
    def get_actual_coordinates(self) -> np.ndarray:
        p1 = self.p1.actual_coordinate()
        p2 = self.p2.actual_coordinate()
        p3 = self.p3.actual_coordinate()
        p4 = self.p4.actual_coordinate()
        return np.array([p1, p2, p3, p4])

    def get_point_list(self) -> np.ndarray:
        return np.array([self.p1.get_coordinates(), self.p2.get_coordinates(),
                         self.p3.get_coordinates(), self.p4.get_coordinates()],
                         dtype=np.float32)

    def __str__(self) -> str:
        return f"{self.p1}, {self.p2}, {self.p3}, {self.p4}"
    

def update_coordinates(event):
    p = Point(event.x, event.y)
    p2 = p.translate_coordinate()
    coordinates_label.config(text=f"X: {p2.x}, Y: {p2.y}")
    if drawing:
        draw_temp_rectangle(begin_point[0], begin_point[1], p.x, p.y)


def start_drawing(event):
    global begin_point, drawing
    begin_point = (event.x, event.y)
    drawing = True


def end_drawing(event):
    global drawing
    canvas.delete("temp_quadrangle")
    draw_rectangle(begin_point[0], begin_point[1], event.x, event.y)
    drawing = False


def draw_temp_rectangle(x1, y1, x2, y2):
    canvas.delete("temp_quadrangle")
    canvas.create_rectangle(
        x1, y1, x2, y2, outline="green", tags="temp_quadrangle")


def draw_rectangle(x1, y1, x2, y2):
    x_mn, x_mx, y_mn, y_mx = min(x1, x2), max(x1, x2), max(y1, y2), min(y1, y2)
    p1 = Point(x_mn, y_mx).translate_coordinate()  # top left
    p2 = Point(x_mx, y_mx).translate_coordinate()  # top right
    p3 = Point(x_mx, y_mn).translate_coordinate()  # bottom right
    p4 = Point(x_mn, y_mn).translate_coordinate()  # bottom left
    quadrangle.append(Quadrangle(p1, p2, p3, p4))
    update_quadrangle_list()
    canvas.create_rectangle(x1, y1, x2, y2, outline="green", tags=f"quadrangle{len(quadrangle)}")
    canvas.create_text(x1+10, y2+10, text=f"Q{len(quadrangle)}", fill="black", tags=f"quadrangle_name{len(quadrangle)}")


def update_quadrangle_list():
    quadrangle_listbox.delete(0, tk.END)
    for i in range(len(quadrangle)):
        quadrangle_listbox.insert(tk.END, f"    Quadrangle_{i+1}: {quadrangle[i].__str__()}")


def show_transform_window():
    def do_transformations():
        try:
            id = int(id_entry.get()) - 1
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
            return
        
        if id < 0 or id >= len(quadrangle):
            messagebox.showerror("Invalid Input", "Id out of range.")
            return
        
        try:
            x = float(translateX_entry.get())
            y = float(translateY_entry.get())
            translate_mat = translating_matrix(Vector(x, y))
            theta_degrees = float(rotate_entry.get())
            theta_radians = degrees_to_radians(theta_degrees)
            rotate_mat = rotation_matrix(theta_radians)
            x = float(scaleX_entry.get())
            y = float(scaleY_entry.get())
            scale_mat = scaling_matrix(Vector(x, y))

            final_mat = translate_mat.dot(rotate_mat).dot(scale_mat)
            transformed_quadrangle = quadrangle[id].transform(final_mat)
            quadrangle.append(transformed_quadrangle)
            update_quadrangle_list()

            p1, p2, p3, p4 = transformed_quadrangle.get_actual_coordinates()
            canvas.create_polygon(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y,
                                    outline="green", tags=f"quadrangle{len(quadrangle)}", fill="")
            canvas.create_text(
                p1.x+10, p1.y+10, text=f"Q{len(quadrangle)}", fill="black", tags=f"quadrangle_name{len(quadrangle)}")
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")
            return

        sub_window.destroy()

    if len(quadrangle) == 0:
        messagebox.showerror("Invalid Operation", "Please draw at least 1 quadrangle.")
        return

    sub_window = tk.Toplevel(root)
    sub_window.title("Transformation Options")

    id_label = tk.Label(
        sub_window, text=f"Enter id of the quadrangle in range [{1}, {len(quadrangle)}].", anchor="w")
    id_label.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")
    id_sub_label = tk.Label(sub_window, text="Id:", anchor="w")
    id_sub_label.grid(row=1, column=0, padx=(20, 0), pady=5, sticky="w")
    id_entry = tk.Entry(sub_window, bd=5, width=5)
    id_entry.grid(row=1, column=0, padx=(40, 0), pady=5, sticky="w")

    translate_label = tk.Label(
        sub_window, text="Enter the coordinates of the translation vector. If you do not want to translate, enter vector (0,0).", anchor="w")
    translate_label.grid(row=2, column=0, padx=(20, 0), pady=20, sticky="w")
    translateX_label = tk.Label(sub_window, text="X:")
    translateX_label.grid(row=3, column=0, padx=(20, 0), pady=5, sticky="w")
    translateX_entry = tk.Entry(sub_window, bd=5, width=5)
    translateX_entry.grid(row=3, column=0, padx=(40, 0), pady=5, sticky="w")
    translateY_label = tk.Label(sub_window, text="Y:")
    translateY_label.grid(row=3, column=0, padx=(120, 0), pady=5, sticky="w")
    translateY_entry = tk.Entry(sub_window, bd=5, width=5)
    translateY_entry.grid(row=3, column=0, padx=(140, 0), pady=5, sticky="w")

    rotate_label = tk.Label(
        sub_window, text="Enter the rotation angle (in degrees). If you do not want to rotate, enter angle 0.", anchor="w")
    rotate_label.grid(row=4, column=0, padx=(20, 0), pady=20, sticky="w")
    rotate_sub_label = tk.Label(sub_window, text="\u0398:", anchor="w")
    rotate_sub_label.grid(row=5, column=0, padx=(20, 0), pady=5, sticky="w")
    rotate_entry = tk.Entry(sub_window, bd=5, width=5)
    rotate_entry.grid(row=5, column=0, padx=(40, 0), pady=5, sticky="w")

    scale_label = tk.Label(
        sub_window, text="Enter the scale vector. If you do not want to scale, enter vector (1,1).", anchor="w")
    scale_label.grid(row=6, column=0, padx=(20, 0), pady=20, sticky="w")
    scaleX_label = tk.Label(sub_window, text="X:")
    scaleX_label.grid(row=7, column=0, padx=(20, 0), pady=5, sticky="w")
    scaleX_entry = tk.Entry(sub_window, bd=5, width=5)
    scaleX_entry.grid(row=7, column=0, padx=(40, 0), pady=5, sticky="w")
    scaleY_label = tk.Label(sub_window, text="Y:")
    scaleY_label.grid(row=7, column=0, padx=(120, 0), pady=5, sticky="w")
    scaleY_entry = tk.Entry(sub_window, bd=5, width=5)
    scaleY_entry.grid(row=7, column=0, padx=(140, 0), pady=5, sticky="w")

    sub_window.geometry("700x500")

    # Create a button to do transformations
    finish_button = tk.Button(sub_window, text="Do transformations",
                              command=do_transformations, cursor="hand2")
    finish_button.grid(row=8, column=0, padx=(40, 0), pady=20, sticky="w")


def show_calc_window():
    def display_result(transform_matrix: np.ndarray):
        def apply_transformation():
            try:
                id = int(id_entry.get()) - 1
            except:
                messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
                return
            
            if id < 0 or id >= len(quadrangle):
                messagebox.showerror("Invalid Input", "Id out of range.")
                return
            
            transformed_quadrangle = quadrangle[id].transform(transform_matrix)
            quadrangle.append(transformed_quadrangle)
            update_quadrangle_list()

            p1, p2, p3, p4 = transformed_quadrangle.get_actual_coordinates()
            canvas.create_polygon(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y,
                                    outline="green", tags=f"quadrangle{len(quadrangle)}", fill="")
            canvas.create_text(
                p1.x+10, p1.y+10, text=f"R{len(quadrangle)}", fill="black", tags=f"quadrangle_name{len(quadrangle)}")
            result_window.destroy()

        result_window = tk.Toplevel(frame)
        result_window.title("Perspective Transformation Matrix")

        matrix_label = tk.Label(result_window, text="Perspective Transformation Matrix:", anchor="w")
        matrix_label.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")

        rows, cols = len(transform_matrix), len(transform_matrix[0])
        for i in range(rows):
            for j in range(cols):
                value = transform_matrix[i][j]
                if abs(value) < 1e-6:
                    value = 0
                label = tk.Label(result_window, text=f"{value:.2f}", width=8, height=4, relief="solid")
                label.grid(row=i+1, column=0, padx=(200*(j+1), 0), pady=5)

        current_row = rows + 1
        id_label = tk.Label(
            result_window, text=f"Enter id of the quadrangle (in range [{1}, {len(quadrangle)}]) to apply this transformation:", anchor="w")
        id_label.grid(row=current_row, column=0, padx=(20, 0), pady=20, sticky="w")
        id_sub_label = tk.Label(result_window, text="Id:", anchor="w")
        id_sub_label.grid(row=current_row + 1, column=0, padx=(20, 0), pady=5, sticky="w")
        id_entry = tk.Entry(result_window, bd=5, width=5)
        id_entry.grid(row=current_row + 1, column=0, padx=(50, 0), pady=5, sticky="w")

        apply_button = tk.Button(result_window, text="Apply",
                              command=apply_transformation, cursor="hand2")
        apply_button.grid(row=current_row + 1, column=0, padx=(120, 0), pady=20, sticky="w")


        result_window.geometry("1000x500")

    def calc_matrices():
        try:
            id1 = int(id1_entry.get()) - 1
            id2 = int(id2_entry.get()) - 1
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid integer value.")
            return
        
        if id1 < 0 or id1 >= len(quadrangle) or id2 < 0 or id2 >= len(quadrangle):
            messagebox.showerror("Invalid Input", "Id out of range.")
            return

        src_quad = quadrangle[id1].get_point_list()
        dst_quad = quadrangle[id2].get_point_list()
        transform_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

        sub_window.destroy()
        if np.linalg.det(transform_matrix) == 0:
            messagebox.showwarning("Failed", f"Cannot transform from the {id1+1} quadrangle to the {id2+1} quadrangle!")
            return
        
        display_result(transform_matrix)


    if len(quadrangle) < 2:
        messagebox.showerror("Invalid Operation", "Please draw at least 2 quadrangles.")
        return

    sub_window = tk.Toplevel(root)
    sub_window.title("Calculate the Perspective Transformation Matrices")

    id1_label = tk.Label(
        sub_window, text=f"Enter id of the original quadrangle in range [{1}, {len(quadrangle)}].", anchor="w")
    id1_label.grid(row=0, column=0, padx=(20, 0), pady=20, sticky="w")
    id1_sub_label = tk.Label(sub_window, text="Id 1:", anchor="w")
    id1_sub_label.grid(row=1, column=0, padx=(20, 0), pady=5, sticky="w")
    id1_entry = tk.Entry(sub_window, bd=5, width=5)
    id1_entry.grid(row=1, column=0, padx=(60, 0), pady=5, sticky="w")

    id2_label = tk.Label(
        sub_window, text=f"Enter id of the target quadrangle in range [{1}, {len(quadrangle)}].", anchor="w")
    id2_label.grid(row=2, column=0, padx=(20, 0), pady=20, sticky="w")
    id2_sub_label = tk.Label(sub_window, text="Id 2:", anchor="w")
    id2_sub_label.grid(row=3, column=0, padx=(20, 0), pady=5, sticky="w")
    id2_entry = tk.Entry(sub_window, bd=5, width=5)
    id2_entry.grid(row=3, column=0, padx=(60, 0), pady=5, sticky="w")

    sub_window.geometry("700x500")

    # Create a button to do calculate the perspective transformation matrices
    finish_button = tk.Button(sub_window, text="Calculate",
                              command=calc_matrices, cursor="hand2")
    finish_button.grid(row=8, column=0, padx=(60, 0), pady=20, sticky="w")


def degrees_to_radians(angle: float) -> float:
    return angle * np.pi / 180


def translating_matrix(v: Vector) -> np.ndarray:
    return np.array([
        [1, 0, v.x],
        [0, 1, v.y],
        [0, 0,  1 ]
    ])


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [      0      ,       0       , 1]
    ])


def scaling_matrix(v: Vector) -> np.ndarray:
    return np.array([
        [v.x,  0,  0],
        [ 0 , v.y, 0],
        [ 0 ,  0,  1]
    ])
    

def scroll_listbox(*args):
    quadrangle_listbox.yview(*args)


def add_new_rect():
    try:
        x1 = float(addX1_entry.get())
        y1 = float(addY1_entry.get())
        x2 = float(addX2_entry.get())
        y2 = float(addY2_entry.get())
        p1 = Point(x1, y1).actual_coordinate()
        p2 = Point(x2, y2).actual_coordinate()
        addX1_entry.delete(0, "end")
        addY1_entry.delete(0, "end")
        addX2_entry.delete(0, "end")
        addY2_entry.delete(0, "end")
        draw_rectangle(p1.x, p1.y, p2.x, p2.y)
    except:
        messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")


def reset_rect():
    for i in range(len(quadrangle)):
        canvas.delete(f"quadrangle{i+1}")
        canvas.delete(f"quadrangle_name{i+1}")
    quadrangle_listbox.delete(0, tk.END)
    quadrangle.clear()


## MAIN ##   

root = tk.Tk()
root.title("Transformation Geometry App")

frame = tk.Frame(root)
frame.grid(row=0, column=0, padx=70, pady=70)

coordinates_label = tk.Label(frame, text="X: 0, Y: 0")
coordinates_label.grid(row=2, column=0)

canvas = tk.Canvas(frame, bg="white", width=750, height=750)
canvas.grid(row=0, column=0, rowspan=2, padx=10,  pady=10)

# Draw horizontal coordinate axis (Ox)
canvas.create_line(0, 380, 750, 380, fill="red", arrow=tk.LAST)
canvas.create_text(740, 370, text="x", fill="red")
canvas.anchor()

# Draw vertical coordinate axis (Oy)
canvas.create_line(380, 0, 380, 750, fill="blue", arrow=tk.FIRST)
canvas.create_text(390, 10, text="y", fill="blue")

# Display the origin
canvas.create_text(370, 390, text="O", fill="black")

# Draw grid representing integral coordinates
for x in range(0, 751, 20):
    canvas.create_line(x, 0, x, 800, fill="gray", dash=(2, 2))
for y in range(0, 751, 20):
    canvas.create_line(0, y, 800, y, fill="gray", dash=(2, 2))


quadrangle = []  # List to store the quadrangles
quadrangle_listbox = tk.Listbox(frame, height=43, width=70)
quadrangle_listbox.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
# Create a Scrollbar
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=scroll_listbox)
scrollbar.grid(row=0, column=2, padx=0, pady=10, sticky="ns")
quadrangle_listbox.config(yscrollcommand=scrollbar.set)

# Form to add new quadrangle
add_quadrangle_label = tk.Label(
    frame, text="Enter the coordinates of two opposite corners of a new rectangle:", anchor="w")
add_quadrangle_label.grid(row=1, column=1, padx=(20, 0), pady=0, sticky="w")
addX1_label = tk.Label(frame, text="X1:")
addX1_label.grid(row=2, column=1, padx=(20, 0), pady=0, sticky="w")
addX1_entry = tk.Entry(frame, bd=5, width=5)
addX1_entry.grid(row=2, column=1, padx=(50, 0), pady=0, sticky="w")
addY1_label = tk.Label(frame, text="Y1:")
addY1_label.grid(row=2, column=1, padx=(120, 0), pady=0, sticky="w")
addY1_entry = tk.Entry(frame, bd=5, width=5)
addY1_entry.grid(row=2, column=1, padx=(150, 0), pady=0, sticky="w")
addX2_label = tk.Label(frame, text="X2:")
addX2_label.grid(row=2, column=1, padx=(220, 0), pady=0, sticky="w")
addX2_entry = tk.Entry(frame, bd=5, width=5)
addX2_entry.grid(row=2, column=1, padx=(250, 0), pady=0, sticky="w")
addY2_label = tk.Label(frame, text="Y2:")
addY2_label.grid(row=2, column=1, padx=(320, 0), pady=0, sticky="w")
addY2_entry = tk.Entry(frame, bd=5, width=5)
addY2_entry.grid(row=2, column=1, padx=(350, 0), pady=0, sticky="w")

add_button = tk.Button(
    frame, text="Add", command=add_new_rect, cursor="hand2")
add_button.grid(row=2, column=1, padx=450, pady=0, sticky="w")

reset_button = tk.Button(
    frame, text="Clear all quadrangles", command=reset_rect, cursor="hand2")
reset_button.grid(row=2, column=1, padx=(550, 0), pady=0, sticky="w")


# Create a button below the Listbox to show the sub-window
transform_button = tk.Button(
    frame, text="Transform a quadrangle", command=show_transform_window, cursor="hand2")
transform_button.grid(row=3, column=1, padx=20, pady=10, sticky="w")

calculate_button = tk.Button(
    frame, text="Calculate transform matrices", command=show_calc_window, cursor="hand2")
calculate_button.grid(row=3, column=1, padx=(220, 0), pady=10, sticky="w")


canvas.bind("<Motion>", update_coordinates)

drawing = False
begin_point = (0, 0)
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<ButtonRelease-1>", end_drawing)

root.mainloop()
