import dufte
import seaborn as sns
import matplotlib.pyplot as plt
import pyvista as pv
import os
import time
dpi = 500

def set_plotting_defaults():
    for k, v in dufte.style.items():
        if "color" in k:
            dufte.style[k] = "black"
    plt.style.use(dufte.style)
    sns.set_context("talk")


def get_screenshot(mesh, filename, cmap="rainbow", scalar="label"):
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh, cmap=cmap, scalars=scalar, show_scalar_bar=False)
    p.camera_position = "yz"
    p.camera.azimuth = 225
    p.camera.elevation = 20
    p.screenshot(filename, transparent_background=True)

