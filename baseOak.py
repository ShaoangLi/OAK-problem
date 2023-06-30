import numpy as np
import math
import random
from scipy.optimize import linprog

def observe_feedbacks(result, c, A, radius, r_sum, c_sum, n_est, n_p):
    for i in range(K):
        if result[i] > 0.5:
            select_arm = i
            for t in range(n_p):
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

def update_estimates(n_est, r_sum, c_sum, r_est, c_est):
    for i in range(K):
        ave = r_sum[i] / max(n_est[i] - 1, 1)
        r_est[i] = min(ave, 1)

        for j in range(d):
            ave = c_sum[j][i] / max(n_est[i] - 1, 1)
            c_est[j][i] = max(ave, 0)

    return r_est, c_est

def update_count(result, delete, basic, epsilon):
    basic_num = 0
    nonbasic_num = 0

    for i in range(len(result)):
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
            delete_count -= 1

def select_suboptimal_arm(reduce, result, count_p, epsilon):
    delete_count = math.ceil(count_p / 4)
    max_values = sorted(reduce, reverse=True)
    for value in max_values:
        if delete_count == 0:
            break
        index = reduce.index(value)
        if result[index] == 1 and reduce[index] > epsilon:
            result[index] = 0
            delete_count -= 1
            Aeq[0][index] = 1000 / epsilon

def base_oak(c, A, b, bounds, T, K, d, Aeq, phase_num, epsilon, radius):
    #initial
    result = [1] * K
    basic = [1] * K
    knapsack = [b] * d
    r_sum = [0] * K
    c_sum = [[0] * K for _ in range(d)]
    r_est = [1] * K
    n_est = [1] * K
    c_est = [[0] * K for _ in range(d)]

    for phase in range(phase_num):
        count = sum(1 for i in range(K) if result[i] > 0.5)
        count_p = sum(1 for i in range(K) if result[i] == 1)
        n_p = int(b * T / (count * phase_num))

        r_sum, c_sum = observe_feedbacks(result, c, A, radius, r_sum, c_sum, n_est, n_p)
        r_est, c_est = update_estimates(n_est, r_sum, c_sum, r_est, c_est)

        r_x = np.array(r_est)
        c_x = np.array(c_est)
        b_x = np.array(knapsack)

        #solve LP
        res = linprog(c=-r_x, A_ub=c_x, b_ub=b_x, bounds=bounds, A_eq=Aeq, b_eq=1)
        res_dual = linprog(c=b_x, A_ub=-c_x.T, b_ub=-r_x, bounds=(0, None), method='simplex')

        #update reduce gap and delete gap
        delete = [0] * K
        reduce = [0] * K

        for i in range(K):
            Aeq[0][i] = 1000 / epsilon
            res_delete = linprog(c=-r_x, A_ub=c_x, b_ub=b_x, bounds=bounds, A_eq=Aeq, b_eq=1)
            Aeq[0][i] = 1
            delete[i] = res_delete.fun - res.fun

        for i in range(K):
            for j in range(d):
                reduce[i] += res_dual.x[j] * c_est[j][i]
            reduce[i] = reduce[i] - r_est[i]

        basic_num, nonbasic_num = update_count(result, delete, basic, epsilon)

        if basic_num >= nonbasic_num:
            select_optimal_arm(delete, result, count_p, epsilon)

        if basic_num <= nonbasic_num:
            select_suboptimal_arm(reduce, result, count_p, epsilon)

    return result


if __name__ == '__main__':
    K = 5
    d = 3
    T = 20000
    phase_num = 5
    epsilon = 0.01 / T


    c = [0.5, 0.45, 0.5, 0.5, 0]
    A = [[0.9, 0.9, 0, 0, 0], [0, 0, 0.9, 0.92, 0], [0.4, 0.4, 0.4, 0.4, 0.4]]
    b = 0.4

    radius = 0.1
    Aeq = [[1, 1, 1, 1, 1]]

    x0_bounds = (0, 1)
    x1_bounds = (0, 1)
    x2_bounds = (0, 1)
    x3_bounds = (0, 1)
    x4_bounds = (0, 1)
    bounds=[x0_bounds, x1_bounds, x2_bounds, x3_bounds,x4_bounds]


    times = 1000
    correct_base = 0
    for ti in range(times):
        result_base = base_oak(c, A, b, bounds, T, K, d, Aeq, phase_num, epsilon, radius)
        if result_base[0] == 2 and result_base[1] == 0 and result_base[2] == 2 and result_base[3] == 0:
            correct_base += 1

    print(correct_base / times)
