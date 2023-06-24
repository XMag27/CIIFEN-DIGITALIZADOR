import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from datetime import datetime, timedelta, time

# Threshold for considering a pixel as "dark" //Base: 125
dark_threshold = 125

# Threshold for considering a pixel as "nearby" //Base: 1.2
distance_threshold = 1.2  # This value may need adjustment

# Minimum number of neighbors //Base: 3
min_neighbors = 3  # Move depending on knn

# Scale for the pluviograph:
scale = 10

# How many pixels per column //Base: 3
# HACE MAS LENTO EL PROCESO, PERO MEJORA LA CALIDAD DE EXTRACCION
pixel_amount = 3

# Load the image in color
img = cv2.imread('test1.tif')

# Upscale
upscale_factor = 3
img = cv2.resize(img, (img.shape[1] * upscale_factor, img.shape[0] * upscale_factor))

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#TODO: Start graph left from first point found!!
#TODO: Superponer la banda para calcar

# Create an empty list to store the coordinates of dark pixels
dark_coords = []

for i in range(gray.shape[1]):
    # Get the indices that would sort the column in ascending order
    sorted_indices = np.argsort(gray[:, i])

    # Get the three darkest indices (where pix int < threshold)
    darkest_indices = sorted_indices[gray[sorted_indices, i] < dark_threshold][:pixel_amount]
    dark_coords.extend([(i, y) for y in darkest_indices])

# Create a KD Tree for efficient neighbor search
tree = cKDTree(dark_coords)

# Filter out dark pixels that do not have a sufficient number of dark neighbors close enough
filtered_dark_coords = []
for coord in dark_coords:
    # Query the tree for neighbors within the distance threshold
    # The result includes the pixel itself, so we subtract 1
    neighbor_count = len(tree.query_ball_point(coord, distance_threshold)) - 1
    if neighbor_count >= min_neighbors:
        filtered_dark_coords.append(coord)

# Print the filtered coordinates
# print(filtered_dark_coords)
# Define the DPI
dpi = 100

# Calculate figure size in inches for 1540 x 353 pixels
figsize = (1540/dpi, 353/dpi)

# Plot the digitalized data for preview
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
points, = ax.plot([], [], 'o', color='red', markersize=1)  # Empty plot for the manual data points

# Create a list to store filtered rainfall data
filtered_rainfall_data = [[] for _ in range(gray.shape[1])]

# Convert filtered dark coordinates to rainfall amounts
for x, y in filtered_dark_coords:
    rainfall_amount = (gray.shape[0] - y) / gray.shape[0] * scale
    filtered_rainfall_data[x].append(rainfall_amount)

# Calculate min of y values for every x and store them
corrected_rainfall_data = [min(column_data) if column_data else None for column_data in filtered_rainfall_data]

# Filter out zeros (None values) before plotting
corrected_rainfall_data = [(x, y) for x, y in enumerate(corrected_rainfall_data) if y is not None]

# Get the lists of x and y values
x_values, y_values = zip(*corrected_rainfall_data)

# Plot the average rainfall data excluding zeros
ax.plot(x_values, y_values, 'o', color='blue', markersize=1)
ax.grid(True)
plt.ylim(0, 11)

# Empty plot for the manual data points
manual_points, = ax.plot([], [], 'o', color='red', markersize=1)

manual_data = []
f = None
combined_data = []
# Boolean variable to control the start and end of mouse dragging
dragging = False

# Variables to hold the selection start and end points
selection_start, selection_end = None, None


def on_press(event):
    global dragging
    if event.button == 1:  # Only start dragging for left mouse button
        dragging = True


def on_release(event):
    global dragging
    if event.button == 1:  # Only stop dragging for left mouse button
        dragging = False


def on_motion(event):
    if dragging and event.button == 1:
        ix, iy = event.xdata, event.ydata
        manual_data.append((ix, iy))
        points.set_data(list(zip(*manual_data)))  # Update the manual data plot
        fig.canvas.draw()
        fig.canvas.flush_events()


def line_select_callback(eclick, erelease):
    'eclick and erelease are matplotlib events at press and release'
    global selection_start, selection_end
    selection_start, selection_end = (eclick.xdata, eclick.ydata), (erelease.xdata, erelease.ydata)


def delete_points(event):
    global corrected_rainfall_data, manual_data, selection_start, selection_end
    if selection_start is not None and selection_end is not None:
        x1, y1 = selection_start if selection_start else (None, None)
        x2, y2 = selection_end if selection_end else (None, None)
        if None not in [x1, y1, x2, y2]:  # Ensure all coordinates are defined
            x1, y1 = min(x1, x2), min(y1, y2)
            x2, y2 = max(x1, x2), max(y1, y2)
            # Delete points from manual_data within the rectangle
            manual_data = [point for point in manual_data if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2)]

            # Delete points from filtered_rainfall_data within the rectangle
            corrected_rainfall_data = [
                [y for idx, y in enumerate(column_data) if not (x1 <= i <= x2 and y1 <= y <= y2)]
                for i, column_data in enumerate(corrected_rainfall_data)
            ]

            if manual_data:  # Only update plot data when manual_data is not empty
                points.set_data(list(zip(*manual_data)))  # Update the manual data plot

    fig.canvas.draw()
    fig.canvas.flush_events()


def delete_all(event):
    global corrected_rainfall_data, manual_data, selection_start, selection_end, points
    if selection_start is not None and selection_end is not None:
        x1, y1 = selection_start if selection_start else (None, None)
        x2, y2 = selection_end if selection_end else (None, None)
        if None not in [x1, y1, x2, y2]:  # Ensure all coordinates are defined
            x1, y1 = min(x1, x2), min(y1, y2)
            x2, y2 = max(x1, x2), max(y1, y2)
            # Delete points from manual_data within the rectangle
            manual_data = [point for point in manual_data if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2)]

            # Delete points from corrected_rainfall_data within the rectangle
            corrected_rainfall_data = [(x, y) for x, y in corrected_rainfall_data if
                                       not (x1 <= x <= x2 and y1 <= y <= y2)]

            # Remove the existing plot lines
            for line in ax.lines:
                line.remove()

            # Redraw the corrected graph
            if corrected_rainfall_data:  # Only plot when corrected_rainfall_data is not empty
                # Get the lists of x and y values
                x_values, y_values = zip(*corrected_rainfall_data)
                ax.plot(x_values, y_values, 'o', color='blue', markersize=1)

            # Recreate the manual points plot
            points, = ax.plot([], [], 'o', color='red', markersize=1)  # Empty plot for the manual data points
            if manual_data:  # Only update plot data when manual_data is not empty
                points.set_data(list(zip(*manual_data)))  # Update the manual data plot

        fig.canvas.draw()
        fig.canvas.flush_events()


def combine_points(corrected_data, manual_data):
    combined_data = corrected_data.copy()  # create a copy of corrected_data
    corrected_x_values = [point[0] for point in corrected_data]  # list of x-values in corrected_data

    for point in manual_data:
        if point[0] not in corrected_x_values:  # check if x-value is in the list of x-values
            combined_data.append(point)  # if not, add the manual point to the end of the combined_data list

    combined_data.sort(key=lambda point: point[0])  # sort combined_data by x-values

    return combined_data


def interpolate_graph(event):
    global combined_data, f
    # Combine the corrected data and manual points
    combined_data = combine_points(corrected_rainfall_data, manual_data)

    # Create x and y arrays from combined_data
    x = np.array([coord[0] for coord in combined_data])
    y = np.array([coord[1] for coord in combined_data])

    # Generate an interpolation function
    f = interp1d(x, y, kind='linear')

    # Create a new figure
    # plt.figure()

    # generate a set of x values covering the range of your original data
    x_values = np.linspace(min(x), max(x), 1000)  # adjust the number of points as needed

    # use your interpolation function to generate y values
    y_values = f(x_values)

    # now you can plot the interpolated data
    # plt.plot(x_values, y_values, '-')
    # plt.show()

    # Plot the combined data as a green line
    ax.plot(x, y, '-', color='green')

    # Plot the data points from corrected_rainfall_data and manual_data over the green line
    x_corrected = np.array([coord[0] for coord in corrected_rainfall_data])
    y_corrected = np.array([coord[1] for coord in corrected_rainfall_data])
    ax.plot(x_corrected, y_corrected, 'o', color='blue', markersize=1)

    if manual_data:
        x_manual, y_manual = zip(*manual_data)
        ax.plot(x_manual, y_manual, 'o', color='red', markersize=1)

    # Redraw the canvas and update the GUI
    fig.canvas.draw()
    fig.canvas.flush_events()


# TODO: Kinda bugged on the x axis for the new graph, revise later.
# TODO: Get source for every point
# TODO: poner intervalo de tiempo

def time_series_calc(f):
    if f is None:
        print('Primero, realiza el AutoFix del grafico.')
    else:
        x_values = f.x
        max_x = max(x_values)
        min_x = np.maximum(min(x_values), 0)

        # Generate new time steps for every 15 minutes over 24 hours
        new_time_steps = np.linspace(min_x, max_x, 96)

        # Use the interpolation function to get the precipitation values at these new time steps
        rainfall_every_15_min = f(new_time_steps)

        # Create a start time at 5:30 AM
        current_time = datetime.combine(datetime.today(), time(5, 30))

        # Prepare a filename with today's date
        filename = datetime.now().strftime('%Y-%m-%d') + '_rainfall_data.txt'

        # Open the file in write mode ('w')
        with open(filename, 'w') as file:

            # Print each time step with its corresponding rainfall value
            for rainfall in rainfall_every_15_min:
                rainfall = round(rainfall, 2)  # Round rainfall to 2 decimal places
                output_str = f"Tiempo: {current_time.strftime('%H:%M')} , Lluvia: {rainfall} mm\n"
                print(output_str)

                # Write the string to the file
                file.write(output_str)

                current_time += timedelta(minutes=15)

        plt.figure(figsize=figsize, dpi=dpi)

        plt.plot([datetime.combine(datetime.today(), time(5, 30)) + timedelta(minutes=15 * i) for i in range(96)],
                 rainfall_every_15_min)

        plt.ylim(0, 11)

        plt.xlabel("Tiempo")
        plt.ylabel("Lluvia (mm)")

        plt.show()


def time_series(event):
    time_series_calc(f)


cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Rectangle selector for deletion
rect = RectangleSelector(ax, line_select_callback, useblit=True,
                         button=[3],  # Right button only
                         minspanx=5, minspany=5,
                         spancoords='pixels',
                         interactive=True)

# Button to delete points
# ax_button_delete = plt.axes([0.905, 0.6, 0.09, 0.05])
# button_delete = Button(ax_button_delete, 'Delete RED')
# button_delete.on_clicked(delete_points)

# Button to delete all
ax_button_delete_all = plt.axes([0.905, 0.5, 0.09, 0.1])
button_delete_all = Button(ax_button_delete_all, 'Del/Reload')
button_delete_all.on_clicked(delete_all)

# Button to autofix graph
ax_button_autofix = plt.axes([0.905, 0.4, 0.09, 0.1])
button_autofix = Button(ax_button_autofix, 'AutoFix')
button_autofix.on_clicked(interpolate_graph)

# Button for timeseries calc:
ax_button_ts = plt.axes([0.905, 0.3, 0.09, 0.1])
button_ts = Button(ax_button_ts, 'TimeSeries')
button_ts.on_clicked(time_series)

# Add labels to the graph
ax.annotate('Lluvia (mm)', xy=(-0.075, 0.5), xycoords='axes fraction',
            xytext=(5, 0), textcoords='offset points',
            rotation=90, ha='left', va='center')
ax.annotate('Tiempo', xy=(0.5, -0.05), xycoords='axes fraction',
            xytext=(0, -10), textcoords='offset points',
            ha='center', va='baseline')

ax.set_title('Azul: Extraccion de scan | Rojo: Correcion manual | Verde: AutoFix')

plt.grid(True)
plt.ylim(0, 11)
plt.show()
