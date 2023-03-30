import main_funks as m
import time
import docx
import matplotlib.pyplot as plt
from config import settings as s

# одиночные эксперименты в сложных местах 

if __name__ == '__main__':
    # 10, 11 элементов G_inh < -0.07

    # Необходимые параметры
    G_inh = 0.07
    s.k_systems = 11
    tMax = 500
    modifier = 'solo'

    # Пути
    path_x_start = s.Graphic_data_path + '_x' + modifier + '.png'
    path_x_end = s.Graphic_data_path + '_x' + modifier + '_end.png'
    path_R = s.Graphic_data_path + '_R' + modifier + '.png'
    path_graph_last_state = s.Graphic_data_path + '_lsFHN_' + modifier + '.png'

    m.generate_file_names_R12_Ginh(modifier=modifier)
    IC = m.generate_IC_any_sizes(dist_between_neurons=s.dist_between_neurons_IC)
    m.plot_IC_FHN(IC, s.pathIC, text='Начальные условия')


    R1_arr, R2_arr, IC, depressed_elements, last_state, [xs, ys, zs, ts] = m.\
        make_experiment(G_inh, IC, tMax, s.highAccuracy, path_graph_x_start=path_x_start, path_graph_x_end=path_x_end, 
        path_graph_R=path_R, path_graph_last_state=path_graph_last_state, do_need_show=False, do_need_xyzt=True)

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
    

