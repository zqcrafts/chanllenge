#  此文件用于合并txt标注，并且将原标记值格式（x_min,y_min,x_max,y_max）改成(x_min,y_min,w,h)

import os

#root = '..\\..\\dataset\\DarkFace_coco_0.666\\train_lable'
#output_path = '..\\..\\dataset\\DarkFace_coco_0.666\\annotations\\train_annotations.txt'
root = '/gdata1/zhuqi/DarkFace_MSRCR_5500/train_lable/'
output_path = '/gdata1/zhuqi/DarkFace_MSRCR_5500/annotations/train_merge.txt'


file_list = os.listdir(root)
file_list.sort(key=lambda x: int(x[:-4]))  # 以文件名倒数第四个数左边（.txt左边）的名称进行排序
print(file_list)

r = open(output_path, 'w', encoding='utf-8-sig')

for i in range(0, len(file_list)):
    print(file_list[i] + '\n')
    r.writelines("#\n")
    r.writelines(file_list[i] + '\n')
    r.writelines("1080 720\n")
    path = os.path.join(root, file_list[i])
    f = open(path, 'r', encoding='utf-8-sig')
    j = 0
    for line in f:
        if j == 0:
            j = 1
            print(line)  # line是逐行读取的字符串
            r.writelines(line)
        else:
            x_min = int(line.split(' ')[0])
            y_min = int(line.split(' ')[1])
            w = int(line.split(' ')[2]) - int(line.split(' ')[0])
            h = int(line.split(' ')[3]) - int(line.split(' ')[1])
            line = str(x_min) + ' ' + str(y_min) + ' ' + str(w) + ' ' + str(h) + ' ' + '1 ' + '\n'

            print(line)  # line是逐行读取的字符串
            r.writelines(line)

