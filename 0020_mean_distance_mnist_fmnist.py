import numpy as np
import matplotlib.pyplot as plt
import dataset as ds


# dataset load
mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()

# get mean data
m_perfects = np.zeros((784,10))
m_num = np.zeros((10,))
f_perfects = np.zeros((784,10))
f_num = np.zeros((10,))

for i in range(len(mtrdx.T)):
    num = np.argmax(mtrdy[:,i])
    m_perfects[:,num] += mtrdx[:,i]
    m_num[num] += 1
all_m_list = []
for i in range(10):
    m_perfects[:,i] /= m_num[i]
    all_m_list.append(m_perfects[:,i].reshape(28,28))
all_m = np.hstack(all_m_list)

for i in range(len(ftrdx.T)):
    num = np.argmax(ftrdy[:,i])
    f_perfects[:,num] += ftrdx[:,i]
    f_num[num] += 1
all_f_list = []
for i in range(10):
    f_perfects[:,i] /= f_num[i]
    all_f_list.append(f_perfects[:,i].reshape(28,28))
all_f = np.hstack(all_f_list)

# L2 norm
m_succ = 0
for i in range(len(mtedx.T)):
    p = np.argmin(np.linalg.norm(m_perfects-mtedx[:,i].reshape(784,1),axis=0))
    m_succ += int(p==np.argmax(mtedy[:,i]))
f_succ = 0
for i in range(len(ftedx.T)):
    p = np.argmin(np.linalg.norm(f_perfects-ftedx[:,i].reshape(784,1),axis=0))
    f_succ += int(p==np.argmax(ftedy[:,i]))

# plot
fig = plt.figure('Teapot: mean images')
fig.suptitle('Mean Images & Right Prediction Ratio')
fig.subplots_adjust(wspace=0,hspace=0,right=1,top=0.9,bottom=0)
gs = fig.add_gridspec(2,2)
label = ('right','wrong')

ax11 = fig.add_subplot(gs[0,0])
ax11.axis('off')
ax11.set_title('mean MNIST')
ax11.imshow(all_m, cmap='gray')

ax12 = fig.add_subplot(gs[0,1])
ax12.pie((m_succ,10000-m_succ),labels=label,explode=(0.1,0),autopct='%1.2f%%')

ax21 = fig.add_subplot(gs[1,0])
ax21.axis('off')
ax21.set_title('mean FMNIST')
ax21.imshow(all_f, cmap='gray')

ax22 = fig.add_subplot(gs[1,1])
ax22.pie((f_succ,10000-f_succ),labels=label,explode=(0.1,0),autopct='%1.2f%%')

plt.show()
