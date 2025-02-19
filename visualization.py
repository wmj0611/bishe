import numpy as np
import matrixGenerator
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.patches import FancyArrowPatch


class SimulationEnvironment:
    def __init__(self):
        self.cameras = []
        self.targets = []

    def add_camera(self, cameras):
        self.cameras = cameras

    def add_target(self, targets):
        self.targets = targets

    def update_visualization(self):
        plt.clf()

        for camera in self.cameras:
            camera_patch = Wedge(
                tuple(camera.position),
                camera.radius,
                camera.azimuth - camera.sector_angle / 2,
                camera.azimuth + camera.sector_angle / 2,
                color='blue',
                alpha=0.3
            )
            plt.gca().add_patch(camera_patch)

            camera_dot = Circle(tuple(camera.position), radius=0.2, color='darkblue')
            plt.gca().add_patch(camera_dot)

        for target in self.targets:
            target_dot = Circle(tuple(target.position), radius=0.1, color='red')
            plt.gca().add_patch(target_dot)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
