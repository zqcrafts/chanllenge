import json
import numpy as np
import matplotlib.pyplot as plt

path = '/gdata2/zhuqi/work_dirs/tinaface/contrast/Diou_Incep/20210403_155428.log.json'

tmp = []
for line in open(path,'r'):
    tmp.append(json.loads(line))

print(type(tmp)) # list[dict1,dict2,...,]  
#{'mode': 'train', 'epoch': 21, 'iter': 78, 'lr': 0.00515, 'memory': 3659, 'data_time': 0.01384, 'loss_cls': 0.22817, 'loss_bbox': 0.63967, 'loss': 0.86785, 'time': 0.37663},
#a = np.array(tmp)
a = tmp
print(type(a))

print(a[1])
print(a[2]['mode'])
print(len(a))

loss = []

for i in range(len(a)):
    if i == 0:
        pass
    elif (a[i]['mode'] == 'train') & (i%20==0):
         loss.append(a[i]['loss'])
    # elif i >= 1:
    #     loss.append(a[i]['loss'])

print(loss)

#x=np.linspace(0,30,len(a)-1)#X轴数据
x=np.linspace(0,30,len(loss))#X轴数据
y1=loss#Y轴数据


#plt.figure(figsize=(8,4))

plt.plot(x,y1,label="$sin(x)$",color="red",linewidth=2)#将$包围的内容渲染为数学公式
plt.show()
plt.savefig(r"./save_test.png",dpi=520)


       

