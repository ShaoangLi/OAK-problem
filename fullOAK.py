import numpy as np
import math
import random
from scipy.optimize import linprog


def first_step_label(c_sum, d, b, T):
    label = 0
    for j in range(d):
        consumption = sum(c_sum[j])
        if consumption > b * T / 2:
            label = 1
    return label


def observe_feedback(n_est, select_arm, r_sum, c_sum, c, A, radius):
    n_est[select_arm] += 1
    if r_sum[select_arm] == 0:
        r_sum[select_arm] += c[select_arm]
    else:
        r_sum[select_arm] += c[select_arm] + random.uniform(-radius, radius)
    for j in range(d):
        if A[j][select_arm] == 0:
            c_sum[j][select_arm] += A[j][select_arm]
        else:
            c_sum[j][select_arm] += A[j][select_arm] + random.uniform(-radius, radius)
    return r_sum, c_sum


def update_first_estimate(K, d, r_sum, c_sum, n_est, r_est, c_est, gamma):
    for i in range(K):
        ave = r_sum[i] / (max(n_est[i] - 1, 1))
        rad = np.sqrt(gamma * ave / n_est[i]) + gamma / n_est[i]
        r_est[i] = min(ave + 2 * rad, 1)

    for j in range(d):
        for i in range(K):
            ave = c_sum[j][i] / (max(n_est[i] - 1, 1))
            rad = np.sqrt(gamma * ave / n_est[i]) + gamma / n_est[i]
            if n_est[i] < 5:
                c_est[j][i] = max(ave - 2 * rad, 0.1)
            else:
                c_est[j][i] = max(ave - 2 * rad, 0)

    return r_est, c_est


def update_second_estimate(result, c, A, radius, r_sum, c_sum, n_est, n_p, r_est, c_est):
    for i in range(len(result)):
        if result[i] > 0.5:
            select_arm = i
            for _ in range(n_p):
                r_sum, c_sum = observe_feedback(n_est, select_arm, r_sum, c_sum, c, A, radius)

            ave = r_sum[i] / (max(n_est[i] - 1, 1))
            r_est[i] = min(ave, 1)

            for j in range(d):
                ave = c_sum[j][i] / (max(n_est[i] - 1, 1))
                c_est[j][i] = max(ave, 0)
    return r_sum, c_sum, r_est, c_est


def update_count(result, delete, basic, epsilon):
    basic_num = 0
    nonbasic_num = 0

    for i in range(K):
        if delete[i] > 0 and result[i] == 1:
            basic[i] = 1
            basic_num += 1
        if delete[i] <= 0 + epsilon and result[i] == 1:
            basic[i] = 0
            nonbasic_num += 1

    return basic_num, nonbasic_num


def select_optimal_arm(delete, result, count_p, epsilon):
    delete_count = math.ceil(count_p / 4)
    max_values = sorted(delete, reverse=True)

    for value in max_values:
        if delete_count == 0:
            break
        index = delete.index(value)
        if result[index] == 1 and delete[index] > epsilon:
            result[index] = 2
            delete_count = delete_count - 1


def select_suboptimal_arm(reduce, result, count_p, epsilon):
    delete_count = math.ceil(count_p / 4)
    max_values = sorted(reduce, reverse=True)
    for value in max_values:
        if delete_count == 0:
            break
        index = reduce.index(value)
        if result[index] == 1 and reduce[index] > epsilon:
            result[index] = 0
            delete_count = delete_count - 1
            Aeq[0][index] = 1000 / epsilon


def full_oak(c, A, b, bounds, T, K, d, Aeq, p, epsilon, radius, gamma, epsilon_b):
    # initial
    result = [1] * K
    basic = [1] * K
    r_sum = [0] * K
    c_sum = [[0] * K for _ in range(d)]
    r_est = [1] * K
    n_est = [1] * K
    c_est = [[0] * K for _ in range(d)]

    knapsack = [(1 - epsilon_b) * b] * d
    dis = [0] * K

    #first step, quite slow
    for t in range(int(T/2)):
        #Allocate appropriate resources
        label = first_step_label(c_sum, d, b, T)
        if label == 1:
            break

        r_est, c_est = update_first_estimate(K, d, r_sum, c_sum, n_est, r_est, c_est, gamma)
        r_x = np.array(r_est)

        res_1 = linprog(c=-r_x, A_ub=c_est, b_ub=knapsack, bounds=bounds, A_eq=Aeq, b_eq=1)
        distribution = res_1.x
        total = np.sum(distribution)
        ratios = distribution / total
        ratios = np.nan_to_num(ratios)
        normalized_ratios = ratios / np.sum(ratios)
        select_arm = np.random.choice(K, p=normalized_ratios)

        r_sum, c_sum = observe_feedback(n_est, select_arm, r_sum, c_sum, c, A, radius)

    #second step
    for phase in range(p):
        count = 0
        count_p = 0
        for i in range(K):
            if result[i] > 0.5:
                if result[i] == 1:
                    count_p += 1
                count += 1
        n_p = int(b * T / (2 * count * p))

        r_sum, c_sum, r_est, c_est = update_second_estimate(result, c, A, radius, r_sum, c_sum, n_est, n_p, r_est, c_est)

        r_x = np.array(r_est)
        c_x = np.array(c_est)
        b_x = np.array(knapsack)

        res = linprog(c=-r_x, A_ub=c_x, b_ub=b_x, bounds=bounds, A_eq=Aeq, b_eq=1)
        res_dual = linprog(c=b_x, A_ub=-c_x.T, b_ub=-r_x, bounds=(0, None), method='simplex')

        delete = [0] * K
        reduce = [0] * K

        for i in range(K):
            Aeq[0][i] = 1000 / epsilon
            res_delete = linprog(c=-r_x, A_ub=c_x, b_ub=b_x, bounds=bounds, A_eq=Aeq, b_eq=1)
            Aeq[0][i] = 1
            delete[i] = res_delete.fun - res.fun

        basic_num, nonbasic_num = update_count(result, delete, basic, epsilon)

        if basic_num >= nonbasic_num:
            select_optimal_arm(delete, result, count_p, epsilon)

        if basic_num <= nonbasic_num:
            for i in range(K):
                for j in range(d):
                    reduce[i] += res_dual.x[j] * c_est[j][i]
                reduce[i] = reduce[i] - r_est[i]
            select_suboptimal_arm(reduce, result, count_p, epsilon)
    return result


if __name__ == '__main__':
    K = 5
    d = 3
    T = 20000
    phase_num = 5
    epsilon = 0.01 / T
    gamma = 2
    epsilon_b = 0.00001
    #choose appropriate parameters

    c = [0.5, 0.49, 0.5, 0.5, 0]
    A = [[0.9, 0.9, 0, 0, 0], [0, 0, 0.9, 0.91, 0], [0.4, 0.4, 0.4, 0.4, 0.4]]
    b = 0.4
    radius = 0.1
    Aeq = [[1, 1, 1, 1, 1]]
    #a hard instance

    x0_bounds = (0, 1)
    x1_bounds = (0, 1)
    x2_bounds = (0, 1)
    x3_bounds = (0, 1)
    x4_bounds = (0, 1)
    bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds,x4_bounds]

    times = 10
    correct_full = 0
    for ti in range(times):
        result_full = full_oak(c, A, b, bounds, T, K, d, Aeq, phase_num, epsilon, radius, gamma, epsilon_b)
        if result_full[0] == 2 and result_full[1] == 0 and result_full[2] == 2 and result_full[3] == 0:
            correct_full += 1

    print(correct_full / times)