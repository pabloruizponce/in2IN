import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from tqdm import tqdm


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=6, mode='interaction'):
    """
    Function to plot an interaction between two agents in 3D in matplotlib
        :param save_path: path to save the animation
        :param kinematic_tree: kinematic tree of the motion
        :param mp_joints: list of motion data for each agent
        :param title: title of the plot
        :param figsize: size of the figure
        :param fps: frames per second of the animation
        :param radius: radius of the plot
        :param mode: mode of the plot
    """
    matplotlib.use('Agg')

    # Define initial limits of the plot
    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        ax.grid(b=False)

    # Funtion to plot a floor in the animation
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # Create the figure and axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()

    # Offsets and colors
    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    # Store the data for each agent
    mp_data = []
    for i,joints in enumerate(mp_joints):

        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })
        
    def update(index):
        """
        Update function for the matplotlib animation
            :param index: index of the frame
        """
        # Update the progress bar
        bar.update(1)

        # Clear the axis and setting initial parameters
        ax.clear()
        plt.axis('off')
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Plot the floor
        plot_xzPlane(-3, 3, 0, -3, 3)

        # Plot each of the persons in the motion
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                linewidth = 3.0
                ax.plot3D(data["joints"][index, chain, 0], 
                          data["joints"][index, chain, 1], 
                          data["joints"][index, chain, 2], 
                          linewidth=linewidth,
                          color=color,
                          alpha=1)

    # Generate animation
    frame_number = min([data.shape[0] for data in mp_joints])
    bar = tqdm(total=frame_number+1)
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()

