from config import settings as s
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
import time
from random import uniform
import memory_work as mem

# Два глобальных параметра
G_inh = 0
tMax = 0

# Start params
a = s.a
b = s.b
S = s.S
tau1 = s.tau1
tau2 = s.tau2
tau3 = s.tau3
V_inh = s.V_inh
V_ex = s.V_ex
k_systems = s.k_systems
k = s.k
G_ex = s.G_ex


def naguma_systems(t, r, k_systems, a_arr):
    global G_inh

    # For make g_ex = array(0)
    # and g_inh = full graph with value G_inh in all
    g_inh = []
    for i in range(0, k_systems):
        g_inh.append([])
        for j in range(0, k_systems):
            if j == i:
                g_inh[i].append(0.0)
            else:
                g_inh[i].append(G_inh)

    res_arr = []
    for i in range(0, k_systems):

        #
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z1_i = r[i*k + 2]
        # z2_i = r[i*k + 3]

        # append fx
        res_arr.append((r[i * k] - r[i * k] ** 3 / 3.0 - r[i * k + 1] - r[i * k + 2] * (r[i * k] - V_inh) + S) / tau1)
        # append fy
        res_arr.append(r[i * k] - b * r[i * k + 1] + a_arr[i])

        # sum fz
        sum_Fz1 = 0.0
        for n in range(0, k_systems):
            sum_Fz1 += g_inh[i][n] * np.heaviside(r[k * n], 0.0)
        # append fz1
        res_arr.append((sum_Fz1 - r[i * k + 2]) / tau2)

    return res_arr

from datetime import datetime
def solve(initial_conditions, a = s.a):
    global G_inh, tMax
    global k_systems
    k_systems = s.k_systems

    # Делаем из а массив
    a_arr : list
    if isinstance(a, list) == False:
        a_arr = [a for i in range(k_systems)]
    else:
        a_arr = a

    # Численное интегрирование
    start_time = time.time()
    sol = 0
    if s.highAccuracy:
        sol = solve_ivp(naguma_systems, [0, tMax], initial_conditions, args=(k_systems, a_arr), rtol=1e-11, atol=1e-11)
    else:
        sol = solve_ivp(naguma_systems, [0, tMax], initial_conditions, args=(k_systems, a_arr),rtol=1e-8, atol=1e-8)
    xs = []
    ys = []
    z1s = []

    ts = sol.t

    for i in range(0, k_systems):
        xs.append(sol.y[i * k])
        ys.append(sol.y[i * k + 1])
        z1s.append(sol.y[i * k + 2])

    print('g_inh: ', G_inh, '\t solve time: ', time.time() - start_time, '\t time: ', datetime.now().time())
    return xs, ys, z1s, ts


def find_maximums(X, t):
    maximums = []
    times_of_maximums = []
    indexes_of_maximums = []
    for i in range(200, len(X) - 1):
        if X[i] > X[i - 1] and X[i] > X[i + 1] and X[i] > 0:
            maximums.append(X[i])
            times_of_maximums.append(t[i])
            indexes_of_maximums.append(i)

    return maximums, times_of_maximums, indexes_of_maximums


def find_period(inform):
    max, time, index = inform

    time_difference = []
    index_difference = []
    for i in range(1, len(max)):
        time_difference.append(time[i] - time[i - 1])
        index_difference.append(index[i] - index[i - 1])
    time_period = sum(time_difference) / len(time_difference)
    index_period = int(sum(index_difference) / len(index_difference))
    return time_period, index_period


def find_period_i(inform, j):
    max, time, index = inform
    return time[j] - time[j - 1], index[j] - index[j - 1]


# Запаздывание между main (первым) нейроном и other(другим) на определенном шаге
def lag_between_neurons(main_t, main_i, other_t, other_i, period, index=3, k_systems=k_systems, x_arr = [], t_arr = []):
    delay = 0
    delay_i = 0
    # Можно попробовать считать период на каждом шаге
    # period2 = main_t[index] - main_t[index - 1]
    # print('period: ', period, 'period 2:', period2)
    try:
        if abs(main_t[index] - other_t[index - 1]) > abs(main_t[index] - other_t[index]):
            delay = other_t[index] - main_t[index]
            delay_i = other_i[index] - main_i[index]
        else:
            delay = other_t[index] - main_t[index - 1]
            delay_i = other_i[index] - main_i[index - 1]
    except:
        print('Опять сломалось')
        
        mem.draw_all_Xt_on_same_graphs(k_systems, x_arr, t_arr)

        print("Len:", len(main_t), len(other_t))
        #print("Broken indexes: ", other_t[index], main_t[index])
        #print("Main time: ", main_t)
        #print("Other time: ", other_t)


    return delay, delay_i


# Подсчет параметра порядка
def find_order_param(period, delays, k_systems_):
    sum_re = 0.0
    sum_im = 0.0
    sum_re2 = 0.0
    sum_im2 = 0.0

    for i in range(0, k_systems_ - 1):
        in_exp = 2 * np.pi * delays[i] / period
        in_exp2 = 4 * np.pi * delays[i] / period
        sum_re += np.cos(in_exp)
        sum_im += np.sin(in_exp)

        sum_re2 += np.cos(in_exp2)
        sum_im2 += np.sin(in_exp2)

    # print('sum_re2', sum_re2, 'sum_im2', sum_im2)
    sum_re += 1.0
    sum_re2 += 1.0
    sum = np.sqrt(sum_re ** 2 + sum_im ** 2)
    sum2 = np.sqrt(sum_re2 ** 2 + sum_im2 ** 2)
    r2 = sum2 / k_systems_
    r = sum / k_systems_
    return r, r2

def find_phases_difference(period, delays, k_systems_):
    phi = []
    for i in range(0, k_systems_ - 1):
        phi_i = 2 * np.pi * delays[i] / period
        phi.append(phi_i)
    return phi


# print начальные условия
def showInitialConditions(IC, k_systems=s.k_systems, name='0'):
    if name == '0':
        print('Initial conditions')
    else:
        print(name)

    for i in range(0, k_systems):
        print(str(IC[i*k]) + ', ' + str(IC[i*k+1]) + ', ' + str(IC[i*k+2]) + ', ')


# Делает solve, рисует x(t) и сохраняет начальные значения в
def solve_and_plot_with_IC(IC, path_graph_x_start=0, path_graph_x_end=0, do_need_show=False, a = s.a):
    margins = {  # +++
        "left": 0.030,
        "bottom": 0.060,
        "right": 0.995,
        "top": 0.950
    }

    # Нужно ли делать принт НУ?
    if do_need_show:
        showInitialConditions(IC)
    xs, ys, z1s, ts = solve(IC, a)

    # Нужно сделать так, чтобы при tMax > 200 рисовалось только последняя часть последовательности
    if (tMax > 200):
        if s.highAccuracy:
            if s.k_systems > 5:
                new_len = 6000
            else:
                new_len = 12000
        else:
            if s.k_systems > 5:
                new_len = 1500
            else:
                new_len = 3000
        short_xs_end = []
        short_ts_end = []
        short_xs_start = []
        short_ts_start = []
        for j in range(0, k_systems):
            short_xs_end.append([])
            short_xs_start.append([])
        for i in range(0, new_len):
            for j in range(0, k_systems):
                short_xs_end[j].append(xs[j][-new_len + i])
                short_xs_start[j].append(xs[j][i])
            short_ts_end.append(ts[-new_len + i])
            short_ts_start.append(ts[i])

        # Осцилограмма на первых точках
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(**margins)
        for i in range(0, k_systems):
            plt.plot(short_ts_start, short_xs_start[i],
                     label=('eq' + str(i + 1)), linestyle=s.plot_styles[i], color=s.plot_colors[i])
            plt.legend()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Осцилограмма x(t) на первых ' + str(new_len) + ' точках')
        plt.ylim(-2.1, 2.1)
        plt.grid()
        # Если передан путь, сохранить график
        if (path_graph_x_start != 0):
            plt.savefig(path_graph_x_start)
        # Нужно ли показывать график
        if do_need_show:
            plt.show()
        plt.close()

        # Осцилограмма на последних точках
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(**margins)
        for i in range(0, k_systems):
            plt.plot(short_ts_end, short_xs_end[i],
                label=('eq' + str(i + 1)), linestyle=s.plot_styles[i], color=s.plot_colors[i])
            plt.legend()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.ylim(-2.1, 2.1)
        plt.title('Осцилограмма x(t) на последних ' + str(new_len) + ' точках')
        plt.grid()

        # Если передан путь, сохранить график
        if (path_graph_x_end != 0):
            plt.savefig(path_graph_x_end)
        # Нужно ли показывать график
        if do_need_show:
            plt.show()
        plt.close()

    else:
        # Полная осцилограмма
        plt.figure(figsize=(30, 5))
        for i in range(0, k_systems):
            plt.plot(ts, xs[i], label=('eq' + str(i + 1)), linestyle=s.plot_styles[i])
            plt.legend()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('Осцилограмма x(t)')
        plt.grid()
        plt.show()

    return xs, ys, z1s, ts


# Отображение начальных условий на единичную окружность
def coords_to_unit_circle(x, y):
    phi = np.arctan(y/x)

    if x < 0:
        phi = phi + np.pi

    # Unit Circle
    x_UC = np.cos(phi)
    y_UC = np.sin(phi)

    return x_UC, y_UC


def generate_your_IC_FHN(arr_indexes_IC, pathIC=0, do_need_show=False):
    if len(arr_indexes_IC) != s.k_systems:
        print('error in generate IC:  len(indexes) = ' + str(len(arr_indexes_IC)) + ', k_systems = ' + str(s.k_systems) )
        return 0
    xs, ys, size = mem.read_FHN_coords_tr()

    for i in range(s.k_systems):
        if arr_indexes_IC[i] >= size or arr_indexes_IC[i] <= -size:
            return 0

    IC = []
    plt.plot(xs, ys)
    for i in range(s.k_systems):
        x = xs[arr_indexes_IC[i]]
        y = ys[arr_indexes_IC[i]]
        plt.scatter(x, y, 200, label=str(i+1), marker=s.scatter_markers[i])

        IC.append(x)
        IC.append(y)
        IC.append(s.z1_IC)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Выбранные начальные условия')
    plt.grid()
    if pathIC:
        plt.savefig(pathIC)
    if do_need_show:
        plt.show()
    plt.close()

    return np.array(IC)


def generate_IC_any_sizes(dist_between_neurons=1, type='prot',
                          do_need_show=False, params_random_generation = (-3, 3, s.pathIC)):
    left_elems = 0
    right_elems = 339
    IC_ind_arr = []

    if type == 'rand':
        return mem.IC_random_generator(params_random_generation[0], params_random_generation[1], pathSave=params_random_generation[2])

    if type == 'unbalanced prot':
        k2 = s.k_systems // 2
        if s.k_systems % 2 == 0:
            type = 'prot'

        else:
            for i in range(0, s.k_systems):
                if i % 2 == 0:
                    IC_ind_arr.append(left_elems - dist_between_neurons * k2 // 2 + dist_between_neurons * i//2)
                else:
                    IC_ind_arr.append(right_elems - dist_between_neurons * k2 // 2 + dist_between_neurons * i)

    if type == 'prot':
        k2 = s.k_systems // 2
        for i in range(0, k2):
            IC_ind_arr.append(left_elems - dist_between_neurons * k2 // 2 + dist_between_neurons * i)
            IC_ind_arr.append(right_elems - dist_between_neurons * k2 // 2 + dist_between_neurons * i)

        if s.k_systems % 2 == 1:
            IC_ind_arr.append((right_elems - left_elems) // 2)

    elif type == 'sol state':
        IC_ind_arr.append(right_elems)
        for i in range(s.k_systems - 1):
            IC_ind_arr.append(left_elems - dist_between_neurons * s.k_systems // 2 + dist_between_neurons * i)
    
    elif type == 'sol state2':
        IC_ind_arr.append(right_elems)
        for i in range(s.k_systems - 1):
            IC_ind_arr.append(left_elems)
    


    elif type == 'full':
        for i in range(0, s.k_systems):
            IC_ind_arr.append(left_elems - dist_between_neurons * s.k_systems // 2 + dist_between_neurons * i)
    
    return generate_your_IC_FHN(IC_ind_arr, do_need_show=do_need_show)
        

# Возвращает True, если подаваемый объект является массивом
def is_list(a):
    if str(type(a)) == "<class 'list'>":
        return True
    return False


# Делает случайный массив с параметром а в диапазоне initial_a +- delta
def make_heterogeneity_a(initial_a, delta, k_systems = s.k_systems):
    eps = 1e-8
    res_arr = []
    for i in range(k_systems):
        res_arr.append(uniform(initial_a - delta + eps, initial_a + delta - eps))
    return res_arr


################################################### make function ######################################################

def make_experiment(G_inh_, IC, tMax_, high_accuracy_=False, path_graph_x_start=0, path_graph_x_end=0,
                    path_graph_R=0, path_graph_last_state=0, path_graph_phi=0, do_need_show=False, do_need_xyzt=False, a=s.a):
    # Две глобальные переменные, которые могут и будут меняться в экспериментах в рамках одного запуска программы
    global tMax, G_inh
    G_inh = G_inh_
    tMax = tMax_

    tMax = tMax_
    s.highAccuracy = high_accuracy_
    k_systems = s.k_systems

    # Получаем численное решение системы
    xs, ys, z1s, ts = solve_and_plot_with_IC(IC, path_graph_x_start, path_graph_x_end, do_need_show = False, a = a)

    # Выбираем eq1 в качестве первого элемента, найдем период его колебаний
    # Трехмерный массив - 1) Номер нейрона; 2) Информация:
    # 0 - координата максимума, 1 - время максимума, 2 - индекс максимума
    inform_about_maximums = []
    for i in range(0, k_systems):
        inform_about_maximums.append(find_maximums(xs[i], ts))
        # print('maximums ' + str(i) + ': ' + str(inform_about_maximums[i][1]))

    # Теперь нужно рассмотреть, не подавлен ли какой элемент
    depressed_elements = []     # список подавленных элементов
    nondepressed_elem = 0
    for i in range(k_systems):
        if len(inform_about_maximums[i][0]) < 10:
            depressed_elements.append(i)
        else:
            nondepressed_elem = i

    # Если подавлены все кроме одного, возвращаем R1,2 = 1 и заканчиваем
    if len(depressed_elements) == k_systems - 1:
        plt.figure()
        len_R = len(inform_about_maximums[nondepressed_elem][2])-2
        R1_arr = np.ones(len_R)
        R2_arr = np.ones(len_R)
        plt.plot(range(len_R), R1_arr, label='R1')
        plt.plot(range(len_R), R2_arr, label='R2')
        plt.title('Зависимость R1, R2 при G_inh = ' + str(G_inh))
        plt.xlabel('k')
        plt.ylabel('R1, R2')
        plt.legend()
        plt.grid()

        # Если передан путь, сохранить изображение
        if (path_graph_R != 0):
            plt.savefig(path_graph_R)

        return R1_arr, R2_arr, IC, depressed_elements

    # Удаляем подавленные элементы
    # 1) Создаем массив xs без подавленных элементов
    xs_no_depressed = []
    for i in range(len(xs)):
        xs_no_depressed.append(xs[i])
    for i in reversed(depressed_elements):
        xs_no_depressed.pop(i)
    # 2) Меняем k_systems, потому что он используется во множестве функций и
    #  с ним должны быть связаны размеры xs и прочие
    k_systems_with_depressed = k_systems
    k_systems -= len(depressed_elements)
    # 2) пересчитываем inform_about_maximums
    inform_about_maximums = []
    for xs_nd in xs_no_depressed:  
        # print(xs_nd)
        # plt.plot(ts, xs_nd)
        # plt.xlim(0, 100)
        # plt.show()
        # plt.close()
        inform_about_maximums.append(find_maximums(xs_nd, ts))

    # Средние периоды колебаний
    mean_periods = []
    for i in range (k_systems):
        mean_periods.append(find_period(inform_about_maximums[i])[0])
    if s.delta > 0:
        print('Mean periods: ', mean_periods)
    difference_of_mean_periods = max(mean_periods) - min(mean_periods)

    delay = []
    R1_arr = []
    R2_arr = []
    J_arr = []
    # phi_arr - двойной массив размером k(максимумов) * (k_systems-1)
    phi_arr = []
    period = 0
    for j in range(10, len(inform_about_maximums[0][2]) - 15):
        delay_in_for = []
        delay_t = []
        for i in range(1, k_systems):
            # Находим период на текущем шаге
            period, i_period = find_period_i(inform_about_maximums[0], j)

            # Находим задержки на текущем шаге
            d, d_t = lag_between_neurons(inform_about_maximums[0][1], inform_about_maximums[0][2],
                                         inform_about_maximums[i][1], inform_about_maximums[i][2], period, j,
                                         k_systems, xs, ts)
            delay_in_for.append(d)
            delay_t.append(d_t)

        for i in range(0, k_systems - 1):
            # if abs(delay_in_for[i]) > period:
            #     if delay_in_for[i] > period:
            #         delay_in_for[i] -= period
            #     elif delay_in_for[i] < period:
            #         delay_in_for[i] += period
            #     i = 0
            delay_in_for[i] = delay_in_for[i] % period
        delay.append(delay_in_for)

        # Находим параметр порядка
        R1, R2 = find_order_param(period, delay_in_for, k_systems)
        R1_arr.append(R1)
        R2_arr.append(R2)
        J_arr.append(j)

        # Находим разности фаз относительно первого элемента
        phi_k = find_phases_difference(period, delay_in_for, k_systems)
        phi_arr.append(phi_k)

    # Транспонируем матрицу разностей фаз и рисуем графики
    phi_arr = np.array(phi_arr)
    phi_arr = phi_arr.T
    

    # Графики параметров порядка
    mem.draw_R12_graphs(G_inh, J_arr, R1_arr, R2_arr, path_graph_R, do_need_show)

    # График разности фаз
    if path_graph_phi:
        mem.draw_phases_graph(J_arr, phi_arr, path_graph_phi)

    # Рисуем конесное состояние системы на единичной окружности
    if path_graph_last_state != 0:
        mem.plot_coords_unit_circle(G_inh, delay[-1], period, path_graph_last_state, k_systems=k_systems)
    plt.close()
    
    k_systems = s.k_systems
    # Необходимо сохранить конечное состояние системы для вывода конечного графика
    last_state = []
    for i in range(k_systems):
        last_state.append(xs[i][-1])
        last_state.append(ys[i][-1])
        last_state.append(z1s[i][-1])

    if s.fhn_full_animation:
        mem.make_animation_fhn_trajectory(k_systems, xs, ys, ts)

    if do_need_xyzt:
        return R1_arr, R2_arr, IC, depressed_elements, last_state, [xs, ys, z1s, ts]

    return R1_arr, R2_arr, IC, [depressed_elements, last_state, phi_arr, [mean_periods, difference_of_mean_periods]]