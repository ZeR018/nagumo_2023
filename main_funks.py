from config import settings as s
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import numpy as np
import time
from random import randint, uniform

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


def naguma_systems(t, r):
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
        res_arr.append(r[i * k] - b * r[i * k + 1] + a)

        # sum fz
        sum_Fz1 = 0.0
        for n in range(0, k_systems):
            sum_Fz1 += g_inh[i][n] * np.heaviside(r[k * n], 0.0)
        # append fz1
        res_arr.append((sum_Fz1 - r[i * k + 2]) / tau2)

    return res_arr

from datetime import datetime
def solve(initial_conditions):
    global G_inh, tMax
    global k_systems
    k_systems = s.k_systems

    start_time = time.time()
    sol = 0
    if s.highAccuracy:
        sol = solve_ivp(naguma_systems, [0, tMax], initial_conditions, rtol=1e-11, atol=1e-11)
    else:
        sol = solve_ivp(naguma_systems, [0, tMax], initial_conditions, rtol=1e-8, atol=1e-8)
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


def find_period(inf):
    max, time, index = inf

    time_difference = []
    index_difference = []
    for i in range(1, len(max)):
        time_difference.append(time[i] - time[i - 1])
        index_difference.append(index[i] - index[i - 1])
    time_period = sum(time_difference) / len(time_difference)
    index_period = int(sum(index_difference) / len(index_difference))
    return time_period, index_period


def find_period_i(inf, j):
    max, time, index = inf
    return time[j] - time[j - 1], index[j] - index[j - 1]


# Запаздывание между main (первым) нейроном и other(другим) на определенном шаге
def lag_between_neurons(main_t, main_i, other_t, other_i, period, index=3):
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

        print(len(main_t), len(other_t))
        print(main_t)
        print(other_t)
        print(other_t[index], main_t[index])


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


def IC_random_generator(a, b):
    random_var = []
    for i in range(0, 2 * k_systems):
        random_var.append(uniform(a, b))
    IC_arr = []

    #print('Random IC:')
    for i in range(0, k_systems):
        IC_arr.append(random_var[i])
        IC_arr.append(random_var[i+1])
        IC_arr.append(s.z1_IC)
    return np.array(IC_arr)


# print начальные условия
def showInitialConditions(IC, k_systems=s.k_systems, name='0'):
    if name == '0':
        print('Initial conditions')
    else:
        print(name)

    for i in range(0, k_systems):
        print(str(IC[i*k]) + ', ' + str(IC[i*k+1]) + ', ' + str(IC[i*k+2]) + ', ')


# Делает solve, рисует x(t) и сохраняет начальные значения в
def solve_and_plot_with_IC(IC, path_graph_x_start=0, path_graph_x_end=0, do_need_show=False):
    margins = {  # +++
        "left": 0.030,
        "bottom": 0.060,
        "right": 0.995,
        "top": 0.950
    }

    # Нужно ли делать принт НУ?
    if do_need_show:
        showInitialConditions(IC)
    xs, ys, z1s, ts = solve(IC)

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

    # Полная осцилограмма
    # plt.figure(figsize=(30, 5))
    # for i in range(0, k_systems):
    #     plt.plot(ts, xs[i], label=('eq' + str(i + 1)), linestyle=s.plot_styles[i])
    #     plt.legend()
    # plt.xlabel('t')
    # plt.ylabel('x')
    # plt.title('Осцилограмма x(t)')
    # plt.grid()
    # plt.show()

    return xs, ys, z1s, ts


def recordICAndR(filename, R, IC, G_inh_arr, size):
    # IC - двумерный массив
    # R - двумерный массив
    f = open(filename, 'w')
    f.write(str(k_systems))
    f.write('\n')
    f.write(str(size))
    f.write('\n')
    for i in range(0, size):
        f.write(str(G_inh_arr[i]))
        f.write('\n')
        f.write(str(len(R[i])))
        f.write('\n')
        for j in range(0, len(IC[i])):
            f.write(str(IC[i][j]))
            f.write('\n')
        # f.writelines(str(IC[i]), '\n')
        for j in range(0, len(R[i])):
            f.write(str(R[i][j]))
            f.write('\n')
    f.close()


def exists(path):
    try:
        file = open(path)
    except IOError as e:
        return False
    else:
        return True



def make_FHN_tr(path):
    # Если такого файла еще нет
    if not exists(path):
        global G_inh
        G_inh = 0.02
        # Запоминаем k_systems на всякий случай
        k_systems_temp = s.k_systems
        k_systems = 1

        IC_1_el = np.array([1., 1., 0.01, 0])
        sol = solve_ivp(naguma_systems, [0, 15], IC_1_el, rtol=1e-11, atol=1e-11)

        xs = sol.y[0]
        ys = sol.y[1]
        ts = sol.t

        short_xs = []
        short_ys = []
        short_ts = []

        x_max_arr, t_max_arr, i_max_arr = find_maximums(xs, ts)
        for index in range(i_max_arr[-2], i_max_arr[-1], 1):
            short_xs.append(xs[index])
            short_ys.append(ys[index])
            short_ts.append(ts[index])

        # Длина полученных массивов
        size = len(short_xs)
        # print(len(short_xs))
        # plt.plot(short_ts, short_xs)
        # plt.grid()
        # plt.show()
        # plt.plot(short_xs, short_ys, alpha=0.5)
        # plt.grid()
        # plt.show()

        with open(path, 'w') as f:
            print(size, file=f)
            for i in range(size):
                print(short_xs[i], short_ys[i], file=f)

        # Возвращаем старый k_systems
        k_systems = k_systems_temp
        return short_xs, short_ys, short_ts, size

    else:
        return 0


# Отображение начальных условий на единичную окружность
def coords_to_unit_circle(x, y):
    phi = np.arctan(y/x)

    if x < 0:
        phi = phi + np.pi

    # Unit Circle
    x_UC = np.cos(phi)
    y_UC = np.sin(phi)

    return x_UC, y_UC


# Сгенерировать НУ на единичной окружности
def IC_FHN_random_generator(path, do_need_show=False, pathSave='0'):
    xs = []
    ys = []
    # Если файла нет, создаем
    if not exists(path):
        make_FHN_tr(path)

    IC = []
    xs, ys, size = read_FHN_coords_tr(path)
    for i in range(k_systems):
        # Рандомим координаты с ФХН
        randIndex = randint(0, len(xs)-1)
        x = xs[randIndex]
        y = ys[randIndex]

        # Записываем координаты на предельном цикле ФХН в НУ
        IC.append(x)
        IC.append(y)
        IC.append(s.z1_IC)

    #plot_IC_unit_circle(IC, pathIC)

    # Рисуем НУ на траектории ФХН
    if do_need_show or pathSave != '0':
        plt.plot(xs, ys)
        for i in range(k_systems):
            plt.scatter(IC[i * k], IC[i * k + 1], 150, label=str(i + 1))
        plt.legend()
        plt.grid()

        if pathSave:
            plt.savefig(pathSave)

        if do_need_show:
            plt.show()
        plt.close()

    if do_need_show:
        showInitialConditions(IC)

    return np.array(IC)


def read_FHN_coords_tr(path=s.FHN_tr_path):
    # Читаем из файла все координаты точек FHN
    with open(path, 'r') as f:
        f_data = f.readlines()

    size = int(f_data[0])

    xs = []
    ys = []
    # Выделяем x, y из строк
    for i in range(1, len(f_data)):
        line = f_data[i].split()
        xs.append(float(line[0]))
        ys.append(float(line[1]))
    return xs, ys, size


def plot_IC_FHN(IC, pathIC=0, pathFHN=s.FHN_tr_path, text='0',
                do_need_show=False):
    xs, ys, size = read_FHN_coords_tr(pathFHN)

    plt.plot(xs, ys)
    for i in range(s.k_systems):
        plt.scatter(IC[i*k], IC[i*k+1], 150, label=str(i+1), marker=s.scatter_markers[i])
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()

    if text != '0':
        plt.title(text)

    if pathIC != 0:
        plt.savefig(pathIC)

    if do_need_show:
        plt.show()
    plt.close()
    return 0


# plot НУ на единичной окружности
def plot_IC_unit_circle(IC, pathIC=0):
    #fig, ax = plt.subplots(figsize=(5, 5))
    plt.Circle((0, 0), 1, fill=False)
    for i in range(k_systems):
        x = IC[i*k]
        y = IC[i*k + 1]
        x, y = coords_to_unit_circle(x, y)
        plt.scatter(x, y, 150, label=str(i+1))

    plt.legend()

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    if pathIC != 0:
        plt.savefig(pathIC)

    #plt.show()
    plt.close()

    return 0


# plot итогового состояния на единичной окружности
def plot_last_coords_unit_circle(delays, period, path_coords=0, do_need_show=False, k_systems=s.k_systems):

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_circle = plt.Circle((0, 0), 1, fill=False)

    # Рисуем 1-й элемент
    plt.scatter(1.0, 0, 150, label=str(1))

    # Обходим каждый элемент
    for i in range(k_systems - 1):
        # Угол i-го элемента
        phi_i = 2 * np.pi * delays[i] / period

        x_i = np.cos(phi_i)
        y_i = np.sin(phi_i)

        plt.scatter(x_i, y_i, 150, label=str(i+2), marker=s.scatter_markers[i])

    ax.set_title('Итоговое состояние при G_inh=' + str(G_inh))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.legend()
    ax.add_artist(draw_circle)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    if path_coords != 0:
        fig.savefig(path_coords)
    if do_need_show:
        plt.show()
    plt.close()

    return 0


def generate_your_IC_FHN(arr_indexes_IC, pathIC=0, do_need_show=False):
    if len(arr_indexes_IC) != s.k_systems:
        print('error in generate IC:  len(indexes) = ' + str(len(arr_indexes_IC)) + ', k_systems = ' + str(s.k_systems) )
        return 0
    xs, ys, size = read_FHN_coords_tr()

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


import os
def write_IC(fName, IC):

    path = os.getcwd()
    path += s.save_IC_data_path

    # Если такой директории нет, создаем
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Создать директорию %s не удалось" % path)

    file_path = path + fName

    f = open(file_path, 'w')
    for i in range(int(len(IC) / k)):
        f.write(str(IC[i*k]) + ',' + str(IC[i*k + 1]) + ',' +
                str(IC[i*k+2]) + ',' + '\n')
    f.close()

    return 0


def read_IC(fName):

    path = os.getcwd()
    file_path = path + s.save_IC_data_path + fName

    if not exists(file_path):
        return -1

    IC = []
    with open(file_path) as f:
        for line in f:
            line_split = line.split(',')
            for i in line_split:
                IC.append(float(i))

    return IC


def generate_IC_any_sizes(dist_between_neurons=1, type='prot',
                          do_need_show=False):
    left_elems = 0
    right_elems = 339
    IC_ind_arr = []

    if type == 'unbalanced prot':
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
    elif type == 'full':
        for i in range(0, s.k_systems):
            IC_ind_arr.append(left_elems - dist_between_neurons * s.k_systems // 2 + dist_between_neurons * i)
    
    return generate_your_IC_FHN(IC_ind_arr, do_need_show=do_need_show)


def write_R12_last(G_inh_arr, R1_arr, R2_arr, file_name = s.R12_last_path):
    f = open(file_name, 'w')
    try:
        for i in range(len(R1_arr)):
            f.write(str(G_inh_arr[i]) + ',' + str(R1_arr[i]) + ',' + str(R2_arr[i]) + '\n')
        return 0
    except:
        ValueError


def read_file_G_dep_R12(file_name = s.R12_last_path):
    G_inh_arr = []
    R1_arr = []
    R2_arr = []

    f  = open(file_name, 'r')
    for line in f:
        l = line.split(',')
        G_inh_arr.append(float(l[0]))
        R1_arr.append(float(l[1]))
        R2_arr.append(float(l[2]))

    return G_inh_arr, R1_arr, R2_arr


def draw_R_dep_G(G_inh_arr, R1_arr, R2_arr, G_inh_arr_r = [], R1_arr_r = [], R2_arr_r = [], path=s.graph_R_Ginh_path, 
                 modifier='', do_need_show=False):
    plt.figure()
    plt.xlabel('G_inh')
    plt.ylabel('R\u2081, R\u2082')
    plt.title('Зависимость R\u2081, R\u2082 от G_inh, ' + str(s.k_systems) + ' элементов')
    plt.grid()
    plt.ylim(-0.05, 1.05)
    
    if R1_arr_r != [] and R2_arr_r != []:   
        plt.plot(G_inh_arr, R1_arr, label='R\u2081_left', color='b')
        plt.plot(G_inh_arr, R2_arr, label='R\u2082_left', color='orange')
    
        plt.plot(G_inh_arr_r, R1_arr_r, label='R\u2081_right', color='g')
        plt.plot(G_inh_arr_r, R2_arr_r, label='R\u2082_right', color='r')

        plt.xlim(-0.09, 0.09)
        plt.savefig(path)
        plt.legend()

        if do_need_show:
            plt.show()
        return 0
    
    plt.plot(G_inh_arr, R1_arr, label='R\u2081')
    plt.plot(G_inh_arr, R2_arr, label='R\u2082')

    if modifier == 'pr_full':
        print('full')
        plt.xlim(-0.09, 0.09)
    else:
        if s.G_inh_sign == -1:
            plt.xlim(-0.09, 0.01)
        else:
            plt.xlim(-0.01, 0.09)

    plt.legend()
    plt.savefig(path)
    if do_need_show:
        plt.show()

    
    return 0
        

# Сверяет, есть ли такое имя существуещего файла, если да, добавляет/меняет
# циферку в конце
def find_new_name_to_file(file_name, k_systems, type='txt', modifier='', small_modifier=''):
    if modifier != '':
        new_path = file_name + '_' + modifier + '_' + str(k_systems)
    else:
        new_path = file_name + '_' + str(k_systems)

    if small_modifier !='':
        new_path += '_' + small_modifier + '_0'
    else:
        new_path += '_0'
    i = 0
    while exists(new_path + '.' + type):
        i += 1
        new_path = new_path[:- (len(str(i-1)) + 1)] + '_' + str(i) 

    return new_path + '.' + type


# Функция для генерирования имен/путей для R12(G_inh)
def generate_file_names_R12_Ginh(modifier = '', small_modifier = ''):
    k_systems = s.k_systems

    s.R12_last_path = find_new_name_to_file(s.R12_last_path_default, k_systems, 'txt', modifier, small_modifier)
    s.path_Doc = find_new_name_to_file(s.path_Doc_default, k_systems,  'docx', modifier, small_modifier)
    s.graph_R_Ginh_path = find_new_name_to_file(s.graph_R_Ginh_path_default, k_systems, 'png', modifier, small_modifier)

    # s.R12_last_path = new_path + '.txt'

    # s.path_Doc = "./Data/results/test.docx"
    # s.graph_R_Ginh_path = './Data/graphics/res_graphs/test.png'
    #graph_R_Ginh_path = './Data/graphics/res_graphs/saved_fig_R_Ginh_pr_full_6_r.png'


################################################### make function ######################################################

def make_experiment(G_inh_, IC, tMax_, high_accuracy_=False, path_graph_x_start=0, path_graph_x_end=0,
                    path_graph_R=0, path_graph_last_state=0, do_need_show=False, do_need_xyzt=False):
    # Две глобальные переменные, которые могут и будут меняться в экспериментах в рамках одного запуска программы
    global tMax, G_inh
    G_inh = G_inh_
    tMax = tMax_

    tMax = tMax_
    s.highAccuracy = high_accuracy_
    k_systems = s.k_systems

    xs, ys, z1s, ts = solve_and_plot_with_IC(IC, path_graph_x_start, path_graph_x_end, do_need_show = False)

    # Выбираем eq1 в качестве первого элемента, найдем период его колебаний
    # Трехмерный массив - 1) Номер нейрона; 2) Информация:
    # 1 - координата максимума, 2 - время максимума, 3 - индекс максимума
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

    delay = []
    R1_arr = []
    R2_arr = []
    J_arr = []
    period = 0
    for j in range(10, len(inform_about_maximums[0][2]) - 15):
        delay_in_for = []
        delay_t = []
        for i in range(1, k_systems):
            # Находим период на текущем шаге
            period, i_period = find_period_i(inform_about_maximums[0], j)

            # Находим задержки на текущем шаге
            d, d_t = lag_between_neurons(inform_about_maximums[0][1], inform_about_maximums[0][2],
                                         inform_about_maximums[i][1], inform_about_maximums[i][2], period, j)
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
    # Классификация не рабочая
    #osc_type = classification(period, delay[-1])
    # if do_need_show:
    #     print('delays:', delay[-1], 'T:', period, 'T/4: ', period / 4.0, 'T/2: ', period / 2.0, '3T/4: ',
    #         3.0 * period / 4.0)
        #print('type: ' + osc_type)

    # Графики параметров порядка
    plt.figure()
    plt.plot(J_arr, R1_arr, label='R\u2081')
    plt.plot(J_arr, R2_arr, label='R\u2082')
    plt.title('Зависимость R\u2081, R\u2082 при G_inh = ' + str(G_inh))
    plt.xlabel('k - k-й номер максимума')
    plt.ylabel('R\u2081, R\u2082')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid()

    # Если передан путь, сохранить изображение
    if(path_graph_R != 0):
        plt.savefig(path_graph_R)
    # Надо ли показывать график
    if do_need_show:
        plt.show()
        plt.close()

    # Рисуем конесное состояние системы на единичной окружности
    if path_graph_last_state != 0:
        plot_last_coords_unit_circle(delay[-1], period, path_graph_last_state, k_systems=k_systems)
    plt.close()
    
    k_systems = s.k_systems
    # Необходимо сохранить конечное состояние системы для вывода конечного графика
    last_state = []
    for i in range(k_systems):
        last_state.append(xs[i][-1])
        last_state.append(ys[i][-1])
        last_state.append(z1s[i][-1])

    if do_need_xyzt:
        return R1_arr, R2_arr, IC, depressed_elements, last_state, [xs, ys, z1s, ts]

    return R1_arr, R2_arr, IC, depressed_elements, last_state