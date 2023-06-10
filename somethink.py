import main_funks as m
from config import settings as s
from matplotlib import pyplot as plt
import numpy as np



G_inh_arr, R1_arr, R2_arr = m.read_file_G_dep_R12("./Data/temp/r12_last_statistic_rand_IC_8_50_0.txt")
plt.figure()
plt.hist([R1_arr, R2_arr], color=['blue', 'orange'], bins=np.linspace(0, 1, 20, endpoint=True), label=['R\u2081', 'R\u2082'])
#plt.hist(R2_arr, color='orange', bins=np.linspace(0, 1, 20, endpoint=True))
plt.legend()
plt.grid()
plt.show()
# # G_inh_arr_2, R1_arr_2, R2_arr_2 = m.read_file_G_dep_R12("./Data/temp/r12_last_pr_full_7_right.txt")
# # m.draw_R_dep_G(G_inh_arr, R1_arr, R2_arr, G_inh_arr_2, R1_arr_2, R2_arr_2,
# #                path='./Data/temp/g1.png', modifier='pr_full', do_need_show=True)
# m.draw_R_dep_G(G_inh_arr, R1_arr, R2_arr,
#                path='./Data/temp/g1.png', modifier='pr_full', do_need_show=True)


