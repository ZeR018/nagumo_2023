import main_funks as m
import time
import docx
import joblib
import matplotlib.pyplot as plt
from config import settings as s


# Обертка для легкой работы с параллелем, основные изменения (параметр связи, пути сохранения) можно делать здесь
def ex_Ginh_f(index, IC, divider, do_need_show=False):
    if isinstance(IC, int or bool):
        IC = m.IC_random_generator(-3, 3)
    path_x_start = s.Graphic_data_path + '_x' + str(index) + '.png'
    path_x_end = s.Graphic_data_path + '_x' + str(index) + '_end.png'
    path_R = s.Graphic_data_path + '_R' + str(index) + '.png'
    path_graph_last_state = s.Graphic_data_path + '_lsFHN_' + str(index) + '.png'

    G_inh = s.G_inh_sign * index / float(divider)
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
                        path_graph_R=path_R, path_graph_last_state=path_graph_last_state, do_need_show=do_need_show)

    return R1_arr, R2_arr, IC, path_x_start, path_x_end, path_R, G_inh, depressed_elements, last_state, path_graph_last_state


# В данный момент нужно доработать - нет записи в txt результатов и 
# Не поддерживает новую систему создания путей 
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
    IC = m.generate_IC_any_sizes(dist_between_neurons=s.dist_between_neurons_IC, type=s.IC_type)
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


def make_protyazhka_R12_dep_G_inh_full(start, stop, step, divider=1000):
    # Настраиваем пути 
    s.G_inh_sign = start // abs(start)
    modifier = 'pr_full'
    if step > 0:
        small_modifier = 'left'
    else:
        small_modifier = 'right'
    m.generate_file_names_R12_Ginh(modifier, small_modifier=small_modifier)
    print('files: ', str(s.path_Doc), str(s.R12_last_path))
    r12_last_path = s.R12_last_path

    # Инициализируем
    final_time = time.time()
    # random_IC = m.IC_random_generator(-3, 3)
    R1_arr_last = []
    R2_arr_last = []
    G_inh_arr = []

    # Инициализируем файл doc
    mydoc = docx.Document()

    # Делаем первые НУ (по дефолту противофазные/циклопные)
    IC = m.generate_IC_any_sizes(dist_between_neurons=s.dist_between_neurons_IC, type=s.IC_type)
    m.plot_IC_FHN(IC, s.pathIC, text='Начальные условия')

    mydoc.add_heading("Initial conditions:", 2)
    for j in range(0, s.k_systems):
        mydoc.add_paragraph(str(IC[j * s.k]) + ', ' + str(IC[j * s.k + 1]) + ', ' +
                            str(IC[j * s.k + 2]) + ', ')
    mydoc.add_picture(s.pathIC)
    mydoc.add_page_break()

    for i in range(start, stop, step):
        if i >= 0:
            s.G_inh_sign = 1
        else:
            s.G_inh_sign = -1
        result = ex_Ginh_f(abs(i), IC, divider, False)
        R1_arr_i = result[0]
        R2_arr_i = result[1]
        IC_i = result[2]
        path_x_start = result[3]
        path_x_end = result[4]
        path_R = result[5]
        G_inh_i = result[6]
        depressed_elements_i = result[7]
        last_state_i = result[8]
        path_graph_last_state = result[9]
            
        # Запись в файл. Docx
        mydoc.add_heading('Experiment ' + str(i), 1)
        mydoc.add_picture(path_x_start, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_x_end, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_R, width=docx.shared.Inches(4))

        # Рисуем конечное состояние на единичной окружности
        mydoc.add_picture(path_graph_last_state, width=docx.shared.Inches(4))

        # Записываем подавленные элементы
        if len(depressed_elements_i) != 0:
            print('depressed', depressed_elements_i)
            mydoc.add_heading('Подавленные элементы: ', 3)
            for el in depressed_elements_i:
                mydoc.add_paragraph(str(el))

        mydoc.add_page_break()

        # Сохраняем в док результатов всё что сделали
        mydoc.save(s.path_Doc)

        # Последние элементы массива R1, R2
        R1_arr_last.append(R1_arr_i[-1])
        R2_arr_last.append(R2_arr_i[-1])
        G_inh_arr.append(G_inh_i)

        # Записываем
        m.write_R12_last(G_inh_arr, R1_arr_last, R2_arr_last, r12_last_path)

        # Делаем начальное состояние следующего эксперемента как конечное состояние предыдущего
        IC = last_state_i
    
    # Делаем график зависимости параметров порядка от силы связи
    m.draw_R_dep_G(G_inh_arr, R1_arr_last, R2_arr_last, path=s.graph_R_Ginh_path, modifier=modifier)

    # Добавляем этот график в док
    mydoc.add_heading('Final graph', 1)
    mydoc.add_picture(s.graph_R_Ginh_path, width=docx.shared.Inches(6))
    mydoc.save(s.path_Doc)

    # Save data
    # m.recordICAndR(R_data_path, R1_arr, IC_arr, G_inh_arr, index)
    print('Process end. Final time: ', time.time() - final_time)


def make_protyazhka_R12_dep_G_inh(start, stop, step, divider=1000):
    # Настраиваем пути 
    modifier = 'pr'
    if step > 0:
        small_modifier = 'left'
    else:
        small_modifier = 'right'
    m.generate_file_names_R12_Ginh(modifier, small_modifier)
    print('files: ', str(s.path_Doc), str(s.R12_last_path))

    # Инициализируем
    final_time = time.time()
    # random_IC = m.IC_random_generator(-3, 3)
    R1_arr_last = []
    R2_arr_last = []
    G_inh_arr = []

    # Инициализируем файл doc
    mydoc = docx.Document()

    # Делаем первые НУ (по дефолту противофазные/циклопные)
    IC = m.generate_IC_any_sizes(dist_between_neurons=s.dist_between_neurons_IC, type=s.IC_type)
    m.plot_IC_FHN(IC, s.pathIC, text='Начальные условия')

    mydoc.add_heading("Initial conditions:", 2)
    for j in range(0, s.k_systems):
        mydoc.add_paragraph(str(IC[j * s.k]) + ', ' + str(IC[j * s.k + 1]) + ', ' +
                            str(IC[j * s.k + 2]) + ', ')
    mydoc.add_picture(s.pathIC)
    mydoc.add_page_break()


    for i in range(start, stop, step):
        result = ex_Ginh_f(i, IC, divider,  False)

        R1_arr_i = result[0]
        R2_arr_i = result[1]
        IC_i = result[2]
        path_x_start = result[3]
        path_x_end = result[4]
        path_R = result[5]
        G_inh_i = result[6]
        depressed_elements_i = result[7]
        last_state_i = result[8]
        path_graph_last_state = result[9]

        mydoc.add_heading('Experiment ' + str(i), 1)
        mydoc.add_picture(path_x_start, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_x_end, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_R, width=docx.shared.Inches(5))
        
        # Рисуем конечное состояние на единичной окружности
        mydoc.add_picture(path_graph_last_state, width=docx.shared.Inches(4))

        # Записываем подавленные элементы
        if len(depressed_elements_i) != 0:
            print('depressed', depressed_elements_i)
            mydoc.add_heading('Подавленные элементы: ', 3)
            for el in depressed_elements_i:
                mydoc.add_paragraph(str(el))

        mydoc.add_page_break()

        mydoc.save(s.path_Doc)

        # Последние элементы массива R1, R2
        R1_arr_last.append(R1_arr_i[-1])
        R2_arr_last.append(R2_arr_i[-1])
        G_inh_arr.append(G_inh_i)

        # Записываем 
        m.write_R12_last(G_inh_arr, R1_arr_last, R2_arr_last, s.R12_last_path)

        # Последние координаты становятся начальными условиями
        IC = last_state_i
        

    # Делаем график зависимости параметров порядка от силы связи
    m.draw_R_dep_G(G_inh_arr, R1_arr_last, R2_arr_last, path=s.graph_R_Ginh_path, modifier=modifier)

    # Добавляем этот график в док
    mydoc.add_heading('Final graph', 1)
    mydoc.add_picture(s.graph_R_Ginh_path, width=docx.shared.Inches(6))
    mydoc.save(s.path_Doc)

    # Save data
    # m.recordICAndR(R_data_path, R1_arr, IC_arr, G_inh_arr, index)
    print('Process end. Final time: ', time.time() - final_time)


#make_protyazhka_R12_dep_G_inh(10, 12, 1)

make_protyazhka_R12_dep_G_inh_full(-47, -80, -1, 1000)
# Посмотреть на резкий переход к режиму (4,2) для отрицательной связи и 6 элементов