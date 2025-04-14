import numpy as np
import matrixGenerator
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.patches import FancyArrowPatch


class SimulationEnvironment:
    def __init__(self):
        self.cameras = []
        self.targets = []
        self.covered_targets = set()  # 存储被覆盖的目标点索引
        self.covered_directions = {}  # 存储每个目标点的覆盖方向

    def add_camera(self, cameras):
        self.cameras = cameras

    def add_target(self, targets):
        self.targets = targets

    def set_covered_targets(self, covered_targets):
        self.covered_targets = covered_targets

    def set_covered_directions(self, covered_directions):
        self.covered_directions = covered_directions

    def update_visualization(self):
        plt.clf()

        # 绘制相机
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

        # 绘制目标点
        for i, target in enumerate(self.targets):
            if i in self.covered_targets:
                # 被覆盖的目标点用绿色表示
                target_dot = Circle(tuple(target.position), radius=0.1, color='green')
            else:
                # 未被覆盖的目标点用红色表示
                target_dot = Circle(tuple(target.position), radius=0.1, color='red')
            plt.gca().add_patch(target_dot)

            # 如果目标点有方向覆盖信息，绘制方向指示
            if i in self.covered_directions:
                for direction in self.covered_directions[i]:
                    angle = np.radians(direction)
                    dx = 0.2 * np.cos(angle)
                    dy = 0.2 * np.sin(angle)
                    plt.arrow(target.position[0], target.position[1], dx, dy,
                            head_width=0.1, head_length=0.1, fc='green', ec='green')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()
