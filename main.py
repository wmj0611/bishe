import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
import copy
import matrixGenerator
import visualization

camera_positions = [(0, 0), (3, 3), (3, -3), (-3, 3), (-3, -3)]# 相机位置
targets_positions = matrixGenerator.generate_random_points(50) #目标位置
facing_directions = [0, 60, 120, 180, 240, 300]#相机方向

num_cameras = len(camera_positions)#相机数量
num_targets = len(targets_positions)#目标的数量

radius = 3#半径长度
sector_angle = 45#相机角度
full_coverage_angle = 90#全视图覆盖角度

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


def main():
    # 四维矩阵，四个维度分别代表目标个数，目标方向数，摄像机个数，摄像机可选方向数

    sim_env = visualization.SimulationEnvironment()
    sim_env.add_camera(cameras)
    sim_env.add_target(targets)
    sim_env.update_visualization()

    matrix = matrixGenerator.find_covered_targets2(cameras, targets, 360 // sector_angle)
    print(matrix)

    matrix_copy = copy.deepcopy(matrix)
    result1, countOfTimes1 = algorithm_1(matrix_copy)
    matrix_copy = copy.deepcopy(matrix)
    result2, countOfTimes2 = algorithm_2(matrix_copy)

    M = len(matrix)
    Q = len(matrix[0])
    N = len(matrix[0][0])
    P = len(matrix[0][0][0])
    print(M, Q, N, P)

    print("Algorithm 1 = " + str(result1))
    print("Iterative Times: " + str(countOfTimes1))
    print("Coverage Rate：" + str(countCoverageRate1(result1, len(matrix))))
    print("Algorithm 2 = " + str(result2))
    for i in range(len(result2)):
        print("the " + str(i) + "th")
        print(setOfFullCoveredTarget(result2[0:i + 1], len(matrix), len(matrix[0])))
    print("Iterative Times: " + str(countOfTimes2))
    print("Coverage Rate: " + str(countCoverageRate2(result2, len(matrix), len(matrix[0]))))


if __name__ == "__main__":
    main()
