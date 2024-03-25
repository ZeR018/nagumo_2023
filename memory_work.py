from config import settings as s
from matplotlib import pyplot as plt
import numpy as np
import os
import main_funks as m
from random import randint, uniform
from PIL import Image
from matplotlib.animation import ArtistAnimation


def IC_random_generator(a, b, pathSave='0', pathFHN=s.FHN_tr_path):
    random_var = []
    for i in range(0, 2 * s.k_systems):
        random_var.append(uniform(a, b))
    IC_arr = []

    #print('Random IC:')
    for i in range(0, s.k_systems):
        IC_arr.append(random_var[i])
        IC_arr.append(random_var[i+1])
        IC_arr.append(s.z1_IC)

    if pathSave != '0':
        xs, ys, size = read_FHN_coords_tr(pathFHN)

        plt.plot(xs, ys)
        for i in range(s.k_systems):
            plt.scatter(IC_arr[i*s.k], IC_arr[i*s.k+1])
        plt.grid()
        plt.savefig(pathSave)
    plt.close()


    return np.array(IC_arr)

def recordICAndR(filename, R, IC, G_inh_arr, size):
    # IC - двумерный массив
    # R - двумерный массив
    f = open(filename, 'w')
    f.write(str(s.k_systems))
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
        sol = m.solve_ivp(m.naguma_systems, [0, 15], IC_1_el, rtol=1e-11, atol=1e-11)

        xs = sol.y[0]
        ys = sol.y[1]
        ts = sol.t

        short_xs = []
        short_ys = []
        short_ts = []

        x_max_arr, t_max_arr, i_max_arr = m.find_maximums(xs, ts)
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


# Сгенерировать НУ на единичной окружности
def IC_FHN_random_generator(path, do_need_show=False, pathSave='0'):
    xs = []
    ys = []
    # Если файла нет, создаем
    if not exists(path):
        make_FHN_tr(path)

    IC = []
    xs, ys, size = read_FHN_coords_tr(path)
    for i in range(s.k_systems):
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
        for i in range(s.k_systems):
            plt.scatter(IC[i * s.k], IC[i * s.k + 1], 150, label=str(i + 1))
        plt.legend()
        plt.grid()

        if pathSave:
            plt.savefig(pathSave)

        if do_need_show:
            plt.show()
        plt.close()

    if do_need_show:
        m.showInitialConditions(IC)

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
        plt.scatter(IC[i*s.k], IC[i*s.k+1], 150, label=str(i+1), marker=s.scatter_markers[i])
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
    for i in range(s.k_systems):
        x = IC[i*s.k]
        y = IC[i*s.k + 1]
        x, y = m.coords_to_unit_circle(x, y)
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
def plot_coords_unit_circle(G_inh, delays, period, path_coords=0, do_need_show=False, k_systems=s.k_systems, title = 'Итоговое состояние при G_inh = '):

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

    if title == 'Итоговое состояние при G_inh = ':
        title += str(G_inh)

    ax.set_title(title)
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
    for i in range(int(len(IC) / s.k)):
        f.write(str(IC[i*s.k]) + ',' + str(IC[i*s.k + 1]) + ',' +
                str(IC[i*s.k+2]) + ',' + '\n')
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
    plt.close()
    
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


# Графики параметров порядка
def draw_R12_graphs(G_inh, ind_arr, R1_arr, R2_arr, path_graph_R=0, do_need_show=0):
    fig = plt.figure()
    plt.plot(ind_arr, R1_arr, label='R\u2081')
    plt.plot(ind_arr, R2_arr, label='R\u2082')
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

    return fig


# График разности фаз
def draw_phases_graph(ind_arr, phi_arr, path_graph_phi):
    fig = plt.figure()
    for i in range(len(phi_arr)):
        for j in range(len(phi_arr[0])):
            if phi_arr[i][j] > np.pi:
                phi_arr[i][j] = phi_arr[i][j] - 2*np.pi
        plt.plot(ind_arr, phi_arr[i], label=('\u03C6' + str(i)))
    plt.title('Разность фаз между первым элементом и i-м')
    plt.xlabel('k - k-й номер максимума')
    plt.ylabel('\u03C6')
    plt.ylim(-np.pi, np.pi)
    plt.legend()
    plt.grid()
    plt.savefig(path_graph_phi)
    plt.close()

    return fig


# График сумм фаз 
def draw_sum_phases(sum_phi_arr, path_graph_sum_phi=0):
    fig = plt.figure()
    plt.plot(range(len(sum_phi_arr)) ,sum_phi_arr)
    plt.title('Сумма фаз по модулю 2\u03C0')
    plt.xlabel('k - k-й номер максимума')
    plt.ylabel('sum')
    plt.grid()
    plt.savefig(path_graph_sum_phi)
    plt.close()

    return fig
    

def draw_all_Xt_on_same_graphs(k_systems, x_arr, t_arr, path_save = s.all_graphs_save_path):
    print('plot')
    margins = {  # +++
        "left": 0.030,
        "bottom": 0.060,
        "right": 0.995,
        "top": 0.950
    }

    time = t_arr[-1]
    time_i = len(t_arr)
    new_len = 0
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

    for i in range(int(time_i / new_len - 1)):
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(**margins)

        for elem in range(k_systems):
            plt.plot(t_arr[new_len * i : new_len * (i + 1)], x_arr[elem][new_len * i : new_len * (i + 1)], label=('eq' + str(elem + 1)), 
                     linestyle=s.plot_styles[elem], color=s.plot_colors[elem])
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.ylim(-2.1, 2.1)
        plt.title("Осциллограмма на промежутке " +str(round(t_arr[new_len * i], 2)) + " - " + str(round(t_arr[new_len * (i + 1)], 2)))
        plt.savefig(path_save + '/xt/_' + str(i) + '.png')

        plt.close()
 
    return 0

def make_graphs_unit_circle_for_gif(G_inh, delays, period, path = s.unit_circle_graphs_data_path):

    for i in range(len(delays)):
        graph_path = path + '/' + str(i) + '.png'
        plot_coords_unit_circle(G_inh, delays[i], period, graph_path)


    return 0

def make_animation_fhn_trajectory(k_systems, x_arr, y_arr, t_arr, frames_interval = 10, gif_interval = 60, pathFHN=s.FHN_tr_path):

    frames = []
    fig = plt.figure()
    xs, ys, size = read_FHN_coords_tr(pathFHN)
    
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-2.1, 2.1)
    ax.grid()
    num_frames = 0
    for i in range(0, len(t_arr), frames_interval):
        frame = []
        line, = ax.plot(xs, ys, color='blue')
        frame.append(line)

        for elem in range(k_systems):
            point = ax.scatter(x_arr[elem][i], y_arr[elem][i], label=('eq' + str(elem + 1)), 
                     linestyle=s.plot_styles[elem], color=s.plot_colors[elem])
            frame.append(point)

        num_frames += 1
        frames.append(frame)
        plt.close()

    print('num frames:', num_frames)
    gif_path = find_new_name_to_file(s.gif_data_path + '/fhn_animation', k_systems, 'gif')
    animation = ArtistAnimation(
                    fig,                # фигура, где отображается анимация
                    frames,              # кадры
                    interval=gif_interval,        # задержка между кадрами в мс
                    blit=True,          # использовать ли двойную буферизацию
                    repeat=False)       # зацикливать ли анимацию
    
    animation.save(gif_path, writer='pillow')

    return 0
