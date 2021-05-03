import pickle
import numpy as np

pickle_file = open('test.pkl', 'rb')

root_path = 'E:\\Contest\\CVPR_UG2_Challenge\\submit\\res\\'
# pickle.dump(my_list, pickle_file) # 用于保存pkl文件
my_list = pickle.load(pickle_file)
pickle_file.close()

file_name = 0
for j in range(100):#range(len(my_list)):
    print(j)
    r = open(root_path + str(j) + '.txt', 'w', encoding='utf-8')
    for i in range(len(my_list[j][0])):

        if my_list[j][0][i][4] > 0.01:
            print(i)
            x_min = my_list[j][0][i][0]
            y_min = my_list[j][0][i][1]
            w = my_list[j][0][i][2]
            h = my_list[j][0][i][3]
            score = my_list[j][0][i][4]

            r.writelines(str(('%.6f' % x_min)) + ' ')
            r.writelines(str(('%.6f' % y_min)) + ' ')
            r.writelines(str(('%.6f' % w)) + ' ')
            r.writelines(str(('%.6f' % h)) + ' ')
            r.writelines(str(('%.6f' % score)) + '\n')







