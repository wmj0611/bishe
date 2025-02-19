import numpy as np


class Camera:
    def __init__(self, idx, position, radius, azimuth, sector_angle, full_coverage_angle):
        self.idx = idx
        self.position = np.array(position)
        self.radius = radius
        self.azimuth = azimuth
        self.sector_angle = sector_angle
        self.full_coverage_angle = full_coverage_angle

    def update_position(self, new_position):
        self.position = np.array(new_position)

    def is_covered(self, target):
        return np.linalg.norm(target.position - self.position) <= self.radius


class Target:
    def __init__(self, idx, position, facing_directions):
        self.idx = idx
        self.position = np.array(position)
        self.facing_directions = facing_directions

    def update_position(self, new_position):
        self.position = np.array(new_position)


def camera_distance(cameras, index1, index2):
    camera1 = cameras[index1]
    camera2 = cameras[index2]
    return np.sqrt(np.sum((camera1.position - camera2.position) ** 2))


def camera_angle(cameras, index1, index2):
    camera1 = cameras[index1]
    camera2 = cameras[index2]
    angle = np.degrees(np.arctan2(camera2.position[1] - camera1.position[1], camera2.position[0] - camera1.position[0]))
    angle = (angle + 360) % 360
    return angle


def place_cameras(num_cameras, camera_positions, radius, sector_angle, full_coverage_angle):
    cameras = []
    for i in range(num_cameras):
        cameras.append(Camera(i + 1, camera_positions[i], radius, 0, sector_angle, full_coverage_angle))
    return cameras


def place_targets(num_targets, target_positions, facing_directions):
    targets = []
    for i in range(num_targets):
        targets.append(Target(i + 1, target_positions[i], facing_directions))
    return targets


def find_covered_targets2(cameras, targets, Q=8):
    matrix = np.full((len(targets), len(targets[0].facing_directions), len(cameras), Q), 0)
    for k in range(len(targets)):
        for l in range(len(targets[k].facing_directions)):
            for i in range(len(cameras)):
                angle_to_target = np.degrees(
                    np.arctan2(targets[k].position[1] - cameras[i].position[1],
                               targets[k].position[0] - cameras[i].position[0]))
                angle_to_target = (angle_to_target + 360) % 360
                this_azimuth = 360 / Q
                angle_difference = abs(angle_to_target - targets[k].facing_directions[l])
                if cameras[i].is_covered(targets[k]) and angle_difference < cameras[i].full_coverage_angle:
                    for j in range(Q):
                        if this_azimuth * j <= angle_to_target < this_azimuth * (j + 1):
                            matrix[k][l][i][j] = 1
    return matrix


def generate_random_points(n):
    # 生成随机数数组，范围为[-3, 3]
    x_values = np.random.uniform(-2, 2, n)
    y_values = np.random.uniform(-2, 2, n)
    # 组合x和y，形成（x,y）序列
    points = [(x, y) for x, y in zip(x_values, y_values)]
    return points
