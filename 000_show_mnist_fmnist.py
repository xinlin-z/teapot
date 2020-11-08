import random
import numpy as np
import dataset as ds
import matplotlib.pyplot as plt


# dataset load
mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()

# plot mnist
fig_m = plt.figure('Teapot : MNIST random show')
axs = [fig_m.add_subplot(i,4,j+(i-1)*4) 
            for i in range(1,3) for j in range(1,5)]
fig_m.subplots_adjust(hspace=2)

for i in range(len(axs)):
    if i < 4:
        pos = random.randint(0,60000)
        img = mtrdx[:,pos]
        num = np.argmax(mtrdy[:,pos])
    else:
        pos = random.randint(0,10000)
        img = mtedx[:,pos]
        num = np.argmax(mtedy[:,pos])
    axs[i].imshow(img.reshape(28,28), cmap='gray')
    axs[i].set_title(str(num))
    axs[i].axis('off')

# plot fmnist
fig_f = plt.figure('Teapot : FMNIST random show')
axs2 = [fig_f.add_subplot(i,4,j+(i-1)*4) 
            for i in range(1,3) for j in range(1,5)]
fig_f.subplots_adjust(hspace=2)

for i in range(len(axs2)):
    if i < 4:
        pos = random.randint(0,60000)
        img = ftrdx[:,pos]
        num = np.argmax(ftrdy[:,pos])
    else:
        pos = random.randint(0,10000)
        img = ftedx[:,pos]
        num = np.argmax(ftedy[:,pos])
    axs2[i].imshow(img.reshape(28,28), cmap='gray')
    axs2[i].set_title(str(num))
    axs2[i].axis('off')

# show
plt.show()


