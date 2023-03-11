import main_funks as m
import time
import docx
import joblib
import matplotlib.pyplot as plt
from config import settings as s


# Обертка для легкой работы с параллелем, основные изменения (параметр связи, пути сохранения) можно делать здесь
def ex_Ginh_f(index, IC, do_need_show=False):
    if isinstance(IC, int or bool):
        IC = m.IC_random_generator(-3, 3)
    path_x_start = s.Graphic_data_path + '_x' + str(index) + '.png'
    path_x_end = s.Graphic_data_path + '_x' + str(index) + '_end.png'
    path_R = s.Graphic_data_path + '_R' + str(index) + '.png'

    G_inh = s.G_inh_sign * index / 1000.0
    #G_inh = -0.045

    # При маленьких значениях параметра связи берем большое время интегрирования
    # if s.G_inh_sign < 0:
    #     if G_inh >= s.G_inh_sign * 0.01:
    #         tMax = s.tMax1
    #     elif G_inh >= s.G_inh_sign * 0.02:
    #         tMax = s.tMax2
    #     else:
    #         tMax = s.tMax3
    # else:
    #     if G_inh <= s.G_inh_sign * 0.01:
    #         tMax = s.tMax1
    #     elif G_inh <= s.G_inh_sign * 0.02:
    #         tMax = s.tMax2
    #     else:
    #         tMax = s.tMax3
    # Только для протяжки берем так
    if s.G_inh_sign < 0:
        if G_inh >= s.G_inh_sign * 0.001:
            tMax = s.tMax1
        elif G_inh >= s.G_inh_sign * 0.002:
            tMax = s.tMax2
        else:
            tMax = s.tMax3
    else:
        if G_inh <= s.G_inh_sign * 0.001:
            tMax = s.tMax1
        elif G_inh <= s.G_inh_sign * 0.002:
            tMax = s.tMax2
        else:
            tMax = s.tMax3

    print('Experiment ' + str(index), 'tMax ' + str(tMax))
    R1_arr, R2_arr, IC, depressed_elements, last_state = m.\
        make_experiment(G_inh, IC, tMax, s.highAccuracy, path_graph_x_start=path_x_start, path_graph_x_end=path_x_end,
                        path_graph_R=path_R, do_need_show=do_need_show)

    return R1_arr, R2_arr, IC, path_x_start, path_x_end, path_R, G_inh, depressed_elements, last_state

def make_R12_dep_G_inh_experiment():
    # Инициализируем
    final_time = time.time()
    # random_IC = m.IC_random_generator(-3, 3)
    R1_arr = []
    R2_arr = []
    IC_arr = []
    index = 0
    osc_types = []

    R1_arr_last = []
    R2_arr_last = []
    G_inh_arr = []

    # Инициализируем файл doc
    mydoc = docx.Document()

    # Делаем НУ, которые будут одинаковы во всех экспериментах
    IC = m.generate_IC_any_sizes(dist_between_neurons=s.dist_between_neurons_IC)
    m.plot_IC_FHN(IC, s.pathIC, text='Начальные условия')

    mydoc.add_heading("Initial conditions:", 2)
    for j in range(0, s.k_systems):
        mydoc.add_paragraph(str(IC[j * s.k]) + ', ' + str(IC[j * s.k + 1]) + ', ' +
                            str(IC[j * s.k + 2]) + ', ')
    mydoc.add_picture(s.pathIC)
    mydoc.add_page_break()


    if s.n_streams == 20:
        range_param = 4
    if s.n_streams == 10:
        range_param = 8
    if s.n_streams == 1:
        range_param = 80
    for i in range(0, range_param):
        loop_index = i * s.n_streams + 1
        # 0.135 - крайнее значение ингибиторной связи

        # Параллелит
        # если один раз рандом на весь эксп - Random_IC
        # если нужно рандомить каждую итерацию - 0
        existance = joblib.Parallel(n_jobs=s.n_streams)(joblib.delayed(ex_Ginh_f)(k, IC, False)
                                                        for k in range(loop_index, loop_index + s.n_streams))

        # Разбираемся с результатами выполнения каждого потока
        for j in range(0, s.n_streams):
            # Записываем данные с каждого потока
            R1_arr_i = existance[j][0]
            R2_arr_i = existance[j][1]
            IC_i = existance[j][2]
            path_x_start = existance[j][3]
            path_x_end = existance[j][4]
            path_R = existance[j][5]
            G_inh_i = existance[j][6]
            depressed_elements_i = existance[j][7]

            R1_arr.append(R1_arr_i)
            R2_arr.append(R2_arr_i)
            IC_arr.append(IC_i)
            G_inh_arr.append(G_inh_i)

            # Запись в файл. Docx
            mydoc.add_heading('Experiment ' + str(loop_index + j), 1)
            mydoc.add_picture(path_x_start, width=docx.shared.Inches(6.5))
            mydoc.add_picture(path_x_end, width=docx.shared.Inches(6.5))
            mydoc.add_picture(path_R, width=docx.shared.Inches(5))
            #mydoc.add_page_break()
            mydoc.save(s.path_Doc)

            # Последние элементы массива R1, R2
            R1_arr_last.append(R1_arr_i[-1])
            R2_arr_last.append(R2_arr_i[-1])

            if len(depressed_elements_i) != 0:
                print('depressed', depressed_elements_i)
                i = 8
                break

    # Делаем график зависимости параметров порядка от силы связи
    plt.figure()
    plt.plot(G_inh_arr, R1_arr_last, label='R\u2081')
    plt.plot(G_inh_arr, R2_arr_last, label='R\u2082')
    plt.xlabel('G_inh')
    plt.ylabel('R\u2081, R\u2082')
    plt.title('Зависимость R\u2081, R\u2082 от G_inh, ' + str(s.k_systems) + ' элементов')
    plt.legend()
    plt.grid()

    plt.ylim(-0.05, 1.05)
    if s.G_inh_sign < 0:
        plt.xlim(-0.09, 0.01)
    else:
        plt.xlim(-0.01, 0.09)
    plt.savefig(s.graph_R_Ginh_path)
    plt.show()
    # Добавляем этот график в док
    mydoc.add_heading('Final graph', 1)
    mydoc.add_picture(s.graph_R_Ginh_path, width=docx.shared.Inches(5))
    mydoc.save(s.path_Doc)

    # Save data
    # m.recordICAndR(R_data_path, R1_arr, IC_arr, G_inh_arr, index)
    print('Process end. Final time: ', time.time() - final_time)


def make_protyazhka_R12_dep_G_inh():
    # Инициализируем
    final_time = time.time()
    # random_IC = m.IC_random_generator(-3, 3)
    R1_arr_last = []
    R2_arr_last = []
    G_inh_arr = []

    # Инициализируем файл doc
    mydoc = docx.Document()

    # Делаем НУ, которые будут одинаковы во всех экспериментах
    IC = m.generate_IC_any_sizes(dist_between_neurons=s.dist_between_neurons_IC)
    m.plot_IC_FHN(IC, s.pathIC, text='Начальные условия')

    mydoc.add_heading("Initial conditions:", 2)
    for j in range(0, s.k_systems):
        mydoc.add_paragraph(str(IC[j * s.k]) + ', ' + str(IC[j * s.k + 1]) + ', ' +
                            str(IC[j * s.k + 2]) + ', ')
    mydoc.add_picture(s.pathIC)
    mydoc.add_page_break()

    for i in range(0, 80):
        loop_index = i + 1
        print(loop_index)
        result = ex_Ginh_f(loop_index, IC, False)
        R1_arr_i = result[0]
        R2_arr_i = result[1]
        IC_i = result[2]
        path_x_start = result[3]
        path_x_end = result[4]
        path_R = result[5]
        G_inh_i = result[6]
        depressed_elements_i = result[7]
        last_state_i = result[8]
            
        mydoc.add_heading('Experiment ' + str(loop_index), 1)
        mydoc.add_picture(path_x_start, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_x_end, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_R, width=docx.shared.Inches(5))
        
        # Запись в файл. Docx
        # Запись начальных условий на каждом эксперименте
        mydoc.add_heading("Initial conditions:", 2)
        for j in range(0, s.k_systems):
            mydoc.add_paragraph(str(IC_i[j * s.k]) + ', ' + str(IC_i[j * s.k + 1]) + ', ' +
                                str(IC_i[j * s.k + 2]) + ', ')

        # Конечное состояние записываем
        mydoc.add_heading("Last state:", 2)
        for j in range(0, s.k_systems):
            mydoc.add_paragraph(str(last_state_i[j * s.k]) + ', ' + str(last_state_i[j * s.k + 1]) + ', ' +
                                str(last_state_i[j * s.k + 2]) + ', ')
        mydoc.add_page_break()

        mydoc.save(s.path_Doc)

        # Последние элементы массива R1, R2
        R1_arr_last.append(R1_arr_i[-1])
        R2_arr_last.append(R2_arr_i[-1])
        G_inh_arr.append(G_inh_i)

        if len(depressed_elements_i) != 0:
            print('depressed', depressed_elements_i)
            i = 8
            break

        # Делаем начальное состояние следующего эксперемента как конечное состояние предыдущего
        IC = last_state_i
    
    # Делаем график зависимости параметров порядка от силы связи
    plt.figure()
    plt.plot(G_inh_arr, R1_arr_last, label='R\u2081')
    plt.plot(G_inh_arr, R2_arr_last, label='R\u2082')
    plt.xlabel('G_inh')
    plt.ylabel('R\u2081, R\u2082')
    plt.title('Зависимость R\u2081, R\u2082 от G_inh, ' + str(s.k_systems) + ' элементов')
    plt.legend()
    plt.grid()

    plt.ylim(-0.05, 1.05)
    if s.G_inh_sign < 0:
        plt.xlim(-0.09, 0.01)
    else:
        plt.xlim(-0.01, 0.09)
    plt.savefig(s.graph_R_Ginh_path)
    plt.show()
    # Добавляем этот график в док
    mydoc.add_heading('Final graph', 1)
    mydoc.add_picture(s.graph_R_Ginh_path, width=docx.shared.Inches(5))
    mydoc.save(s.path_Doc)

    # Save data
    # m.recordICAndR(R_data_path, R1_arr, IC_arr, G_inh_arr, index)
    print('Process end. Final time: ', time.time() - final_time)



make_protyazhka_R12_dep_G_inh()


# Traceback (most recent call last):
#   File "d:\Уроки\coursework\nagumo_2023\graph_depend_R12_on_G_inh.py", line 247, in <module>
#     make_protyazhka_R12_dep_G_inh()
#   File "d:\Уроки\coursework\nagumo_2023\graph_depend_R12_on_G_inh.py", line 221, in make_protyazhka_R12_dep_G_inh
#     plt.plot(G_inh_arr, R1_arr_last, label='R\u2081')
#   File "C:\Users\Eugene\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\pyplot.py", line 2767, in plot
#     return gca().plot(
#   File "C:\Users\Eugene\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\axes\_axes.py", line 1635, in plot
#     lines = [*self._get_lines(*args, data=data, **kwargs)]
#   File "C:\Users\Eugene\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\axes\_base.py", line 312, in __call__
#     yield from self._plot_args(this, kwargs)
#   File "C:\Users\Eugene\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\axes\_base.py", line 498, in _plot_args
#     raise ValueError(f"x and y must have same first dimension, but "
# ValueError: x and y must have same first dimension, but have shapes (0,) and (80,)
# PS D:\Уроки\coursework\nagumo_2023> 