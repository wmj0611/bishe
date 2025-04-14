import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import copy
import matrixGenerator
import visualization
import matplotlib.pyplot as plt

camera_positions = [(0, 0), (3, 3), (3, -3), (-3, 3), (-3, -3)]  # 相机位置
targets_positions = matrixGenerator.generate_random_points(50)  # 目标位置
facing_directions = [0, 60, 120, 180, 240, 300]  # 相机方向

num_cameras = len(camera_positions)  # 相机数量
num_targets = len(targets_positions)  # 目标的数量

radius = 3  # 半径长度
sector_angle = 45  # 相机角度
full_coverage_angle = 90  # 全视图覆盖角度

cameras = matrixGenerator.place_cameras(num_cameras, camera_positions, radius, sector_angle, full_coverage_angle)
targets = matrixGenerator.place_targets(num_targets, targets_positions, facing_directions)


def solve_MILP1(matrix, alpha=1, beta=0):
    # 矩阵的维度
    M = len(matrix)
    Q = len(matrix[0])
    N = len(matrix[0][0])
    P = len(matrix[0][0][0])

    # 创建PuLP问题
    prob = LpProblem(name="MILP_Problem", sense=LpMaximize)

    # 创建变量
    y = {k: LpVariable(name=f"y_{k}", lowBound=0, upBound=1, cat='Integer') for k in range(M)}
    delta = {(k, l): LpVariable(name=f"delta_{k}_{l}", lowBound=0, cat='Continuous') for k in range(M) for l in
             range(Q)}
    h = {(k, l): LpVariable(name=f"h_{k}_{l}", lowBound=0, upBound=1, cat='Integer') for k in range(M) for l in
         range(Q)}
    x = {(i, j): LpVariable(name=f"x_{i}_{j}", cat='Binary') for i in range(N) for j in range(P)}

    # 定义目标函数
    prob += alpha * lpSum([y[k] for k in range(M)]) + beta * lpSum(
        [lpSum([h[k, l] for l in range(Q)]) for k in range(M)])

    # 添加约束
    for k in range(M):
        for l in range(Q):
            prob += delta[k, l] == lpSum([matrix[k][l][i][j] * x[i, j] for i in range(N) for j in range(P)])
            prob += delta[k, l] * 1.0 / N <= h[k, l]
            prob += h[k, l] <= delta[k, l]

    for k in range(M):
        prob += lpSum([h[k, l] for l in range(Q)]) >= y[k] * Q

    for i in range(N):
        prob += lpSum([x[i, j] for j in range(P)]) <= 1

    # 求解问题
    prob.solve()

    return prob.objective.value(), {v.name: v.value() for v in prob.variables()}


def solve_MILP2(matrix, alpha=3, beta=1):
    # 矩阵的维度
    M = len(matrix)
    Q = len(matrix[0])
    N = len(matrix[0][0])
    P = len(matrix[0][0][0])

    # 创建PuLP问题
    prob = LpProblem(name="MILP_Problem", sense=LpMaximize)

    # 创建变量
    y = {k: LpVariable(name=f"y_{k}", lowBound=0, upBound=1, cat='Integer') for k in range(M)}
    delta = {(k, l): LpVariable(name=f"delta_{k}_{l}", lowBound=0, cat='Continuous') for k in range(M) for l in
             range(Q)}
    h = {(k, l): LpVariable(name=f"h_{k}_{l}", lowBound=0, upBound=1, cat='Integer') for k in range(M) for l in
         range(Q)}
    x = {(i, j): LpVariable(name=f"x_{i}_{j}", cat='Binary') for i in range(N) for j in range(P)}

    # 定义目标函数
    prob += alpha * lpSum([y[k] for k in range(M)]) + beta * lpSum(
        [lpSum([h[k, l] for l in range(Q)]) for k in range(M)])

    # 添加约束
    for k in range(M):
        for l in range(Q):
            prob += delta[k, l] == lpSum([matrix[k][l][i][j] * x[i, j] for i in range(N) for j in range(P)])
            prob += delta[k, l] * 1.0 / N <= h[k, l]
            prob += h[k, l] <= delta[k, l]

    for k in range(M):
        prob += lpSum([h[k, l] for l in range(Q)]) >= y[k] * Q

    for i in range(N):
        prob += lpSum([x[i, j] for j in range(P)]) <= 1

    # 求解问题
    prob.solve()

    return prob.objective.value(), {v.name: v.value() for v in prob.variables()}


def removeTargets(set_of_indices, matrix):
    for index in set_of_indices:
        for j in range(len(matrix[index])):
            for k in range(len(matrix[index][j])):
                for l in range(len(matrix[index][j][k])):
                    matrix[index][j][k][l] = 0
    return matrix


def removeDirection(set_of_indices, matrix):
    for index in set_of_indices:
        i, j = index
        for sublist in matrix[i][j]:
            for k in range(len(sublist)):
                sublist[k] = 0
    return matrix


def algorithm_1(matrix):
    countOfTimes = 0
    result = []
    covered_targets = set()
    while True:
        object_value, variable_values = solve_MILP1(matrix)
        print("Objective value:", object_value)

        M = len(matrix)

        new_covered_targets = set()
        for k in range(M):
            if variable_values["y_" + str(k)] == 1 and k not in covered_targets:
                new_covered_targets.add(k)

        if not new_covered_targets:
            break
        else:
            countOfTimes += 1
            result.append(new_covered_targets)
            removeTargets(new_covered_targets, matrix)

        print("New targets covered:", new_covered_targets)
        covered_targets.update(new_covered_targets)

        print("Final solution:")
        for name, value in variable_values.items():
            print(name, "=", value)
    return result, countOfTimes


def algorithm_2(matrix):
    countOfTimes = 0
    result = []
    covered_targets = set()
    while True:
        object_value, variable_values = solve_MILP2(matrix)
        print("Objective value:", object_value)

        M = len(matrix)
        Q = len(matrix[0])

        flag = 0
        new_covered_targets = set()
        for k in range(M):
            for l in range(Q):
                if variable_values["h_" + str(k) + "_" + str(l)] == 1 and (k, l) not in covered_targets:
                    new_covered_targets.add((k, l))
                # if variable_values["y_" + str(k)] == 1:
                #     flag = 1

        if not new_covered_targets:
            break
        else:
            countOfTimes += 1
            result.append(new_covered_targets)
            print(setOfFullCoveredTarget(result, M, Q))
            removeTargets(setOfFullCoveredTarget(result, M, Q), matrix)

        print("New targets covered:", new_covered_targets)
        covered_targets.update(new_covered_targets)

        print("Final solution:")
        for name, value in variable_values.items():
            print(name, "=", value)
    return result, countOfTimes


def countCoverageRate1(sequence, k):
    count = 0
    for _ in sequence:
        count += len(_)
    return count / k


def printMatrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])


def setOfFullCoveredTarget(sequence, k, n):
    set_of_result = set()
    occurrences = {}

    # 初始化字典，为每个 i 创建一个空集合
    for i in range(k):
        occurrences[i] = set()

    # 遍历序列，更新每个 i 对应的 j 的出现情况
    for s in sequence:
        for i, j in s:
            occurrences[i].add(j)

    # 检查是否同时存在 (i, 0), (i, 1), ..., (i, n - 1)
    for i in range(k):
        if all(j in occurrences[i] for j in range(n)):
            set_of_result.add(i)

    return set_of_result


def countCoverageRate2(sequence, k, n):
    count = len(setOfFullCoveredTarget(sequence, k, n))
    return count / k


def extract_camera_angles(variable_values, num_cameras, num_directions):
    camera_angles = {}
    for i in range(num_cameras):
        for j in range(num_directions):
            if variable_values.get(f"x_{i}_{j}") == 1:
                camera_angles[i] = j * (360 / num_directions)
    return camera_angles


def update_camera_angles(cameras, camera_angles):
    for camera_idx, angle in camera_angles.items():
        cameras[camera_idx].azimuth = angle


def calculate_actual_coverage(cameras, targets):
    """计算实际的覆盖率"""
    covered_targets = set()
    covered_directions = {}

    for i, target in enumerate(targets):
        covered_directions[i] = set()
        for camera in cameras:
            if camera.is_covered(target):
                # 计算目标点到相机的角度
                angle_to_camera = np.degrees(
                    np.arctan2(target.position[1] - camera.position[1],
                               target.position[0] - camera.position[0]))
                angle_to_camera = (angle_to_camera + 360) % 360

                # 检查是否在相机的覆盖范围内
                angle_diff = abs(angle_to_camera - camera.azimuth)
                if angle_diff <= camera.sector_angle / 2:
                    covered_targets.add(i)
                    covered_directions[i].add(angle_to_camera)

    # 计算覆盖率
    coverage_rate = len(covered_targets) / len(targets)

    return coverage_rate, covered_targets, covered_directions


def main():
    # 四维矩阵，四个维度分别代表目标个数，目标方向数，摄像机个数，摄像机可选方向数

    sim_env = visualization.SimulationEnvironment()
    sim_env.add_camera(cameras)
    sim_env.add_target(targets)

    # 计算并显示初始覆盖率
    initial_coverage, covered_targets, covered_directions = calculate_actual_coverage(cameras, targets)
    sim_env.set_covered_targets(covered_targets)
    sim_env.set_covered_directions(covered_directions)
    sim_env.update_visualization()

    matrix = matrixGenerator.find_covered_targets2(cameras, targets, 360 // sector_angle)

    # 对于算法1
    matrix_copy = copy.deepcopy(matrix)
    result1, countOfTimes1 = algorithm_1(matrix_copy)
    object_value1, variable_values1 = solve_MILP1(matrix_copy)

    # 提取相机角度并更新
    camera_angles1 = extract_camera_angles(variable_values1, num_cameras, len(facing_directions))
    cameras_copy1 = copy.deepcopy(cameras)
    update_camera_angles(cameras_copy1, camera_angles1)

    # 计算并显示算法1的实际覆盖率
    coverage1, covered_targets1, covered_directions1 = calculate_actual_coverage(cameras_copy1, targets)
    sim_env1 = visualization.SimulationEnvironment()
    sim_env1.add_camera(cameras_copy1)
    sim_env1.add_target(targets)
    sim_env1.set_covered_targets(covered_targets1)
    sim_env1.set_covered_directions(covered_directions1)
    plt.title(f"Algorithm 1 Result (Coverage: {coverage1:.2%})")
    sim_env1.update_visualization()

    # 对于算法2
    matrix_copy = copy.deepcopy(matrix)
    result2, countOfTimes2 = algorithm_2(matrix_copy)
    object_value2, variable_values2 = solve_MILP2(matrix_copy)

    # 提取相机角度并更新
    camera_angles2 = extract_camera_angles(variable_values2, num_cameras, len(facing_directions))
    cameras_copy2 = copy.deepcopy(cameras)
    update_camera_angles(cameras_copy2, camera_angles2)

    # 计算并显示算法2的实际覆盖率
    coverage2, covered_targets2, covered_directions2 = calculate_actual_coverage(cameras_copy2, targets)
    sim_env2 = visualization.SimulationEnvironment()
    sim_env2.add_camera(cameras_copy2)
    sim_env2.add_target(targets)
    sim_env2.set_covered_targets(covered_targets2)
    sim_env2.set_covered_directions(covered_directions2)
    plt.title(f"Algorithm 2 Result (Coverage: {coverage2:.2%})")
    sim_env2.update_visualization()

    # 只输出算法1和算法2的覆盖率
    print("\nCoverage Results:")
    print(f"Algorithm 1 Coverage Rate: {coverage1:.2%}")
    print(f"Algorithm 2 Coverage Rate: {coverage2:.2%}")


if __name__ == "__main__":
    main()
