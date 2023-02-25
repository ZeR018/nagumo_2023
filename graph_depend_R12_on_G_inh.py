import main_funks as m
import time
import docx
import joblib
import matplotlib.pyplot as plt
from config import settings as s

# Инициализируем
final_time = time.time()
#random_IC = m.IC_random_generator(-3, 3)
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


# Обертка для легкой работы с параллелем, основные изменения (параметр связи, пути сохранения) можно делать здесь
def ex_Ginh_f(index, IC, do_need_show=False):
    if isinstance(IC, int or bool):
        IC = m.IC_random_generator(-3, 3)
    path_x = s.Graphic_data_path + '_x' + str(index) + '.png'
    path_R = s.Graphic_data_path + '_R' + str(index) + '.png'

    G_inh = - index / 1000.0
    #G_inh = -0.045

    # При маленьких значениях параметра связи берем большое время интегрирования
    if G_inh > - 0.005:
        tMax = 10000
    elif G_inh > - 0.02:
        tMax = 5000
    else:
        tMax = 3000

    print('Experiment ' + str(index), 'tMax ' + str(tMax))
    R1_arr, R2_arr, IC, depressed_elements, last_state = m.\
        make_experiment(G_inh, IC, tMax, s.highAccuracy, path_graph_x_end=path_x,
                        path_graph_R=path_R, do_need_show=do_need_show)

    return R1_arr, R2_arr, IC, path_x, path_R, G_inh

# Делаем НУ, которые будут одинаковы во всех экспериментах
left_elems = 0
right_elems = 339
protivofaza_ic = [left_elems, left_elems,
                  right_elems, right_elems]

#IC_10 = [left_elems - 2, left_elems - 1, left_elems, left_elems + 1, left_elems + 2,
#         right_elems - 2, right_elems - 1, right_elems, right_elems + 1, right_elems + 2]
IC_6 = [left_elems - 1, left_elems, left_elems + 1,
        right_elems - 1, right_elems, right_elems + 1]
IC = m.generate_your_IC_FHN(IC_6, do_need_show=False)

mydoc.add_heading("Initial conditions:", 2)
for j in range(0, s.k_systems):
    mydoc.add_paragraph(str(IC[j * s.k]) + ', ' + str(IC[j * s.k + 1]) + ', ' +
                        str(IC[j * s.k + 2]) + ', ')

for i in range(0, 8):
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
        path_x = existance[j][3]
        path_R = existance[j][4]
        G_inh_i = existance[j][5]

        R1_arr.append(R1_arr_i)
        R2_arr.append(R2_arr_i)
        IC_arr.append(IC_i)
        G_inh_arr.append(G_inh_i)

        # Запись в файл. Docx
        mydoc.add_heading('Experiment ' + str(loop_index + j), 1)
        mydoc.add_picture(path_x, width=docx.shared.Inches(6.5))
        mydoc.add_picture(path_R, width=docx.shared.Inches(5))
        mydoc.save(s.path_Doc)

        # Последние элементы массива R1, R2
        R1_arr_last.append(R1_arr_i[-1])
        R2_arr_last.append(R2_arr_i[-1])

    index += 1

# Делаем график зависимости параметров порядка от силы связи
plt.figure()
plt.plot(G_inh_arr, R1_arr_last, label='R\u2081')
plt.plot(G_inh_arr, R2_arr_last, label='R\u2082')
plt.xlabel('G_inh')
plt.ylabel('R\u2081, R\u2082')
plt.title('Зависимость R_1, R_2 от G_inh')
plt.legend()
plt.grid()
plt.savefig(s.graph_R_Ginh_path)
plt.show()
# Добавляем этот график в док
mydoc.add_heading('Final graph', 1)
mydoc.add_picture(s.graph_R_Ginh_path, width=docx.shared.Inches(5))
mydoc.save(s.path_Doc)



# Save data
#m.recordICAndR(R_data_path, R1_arr, IC_arr, G_inh_arr, index)




print('Process end. Final time: ', time.time() - final_time)