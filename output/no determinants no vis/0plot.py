import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Directory containing the .txt files
directory = "."

# Get a list of all .txt files in the directory
file_list = glob.glob(os.path.join(directory, "iteration_*Predecessors.txt"))
if len(file_list) == 0:
    raise ValueError("No files found in the directory")

# Sort the files to ensure they are in the correct order
file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 10))

# Define a function to update the plot for each frame of the animation
def update(frame):
    # Load the data from the current file
    try:
        data = np.loadtxt(file_list[frame])
    except:
        raise ValueError(f"Error loading file {file_list[frame]}")
    # Clear the current plot and create a heatmap of the data
    ax.clear()
    ax.imshow(data, cmap='viridis',)
    ax.set_title(f"Frame {frame+1}")

# Create the animation
anim = FuncAnimation(fig, update, frames=len(file_list), repeat=True, interval = 300)
anim.save('000predecssors.gif', writer='pillow')
# Show the animation
plt.show()