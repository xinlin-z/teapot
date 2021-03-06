import random
import numpy as np
import dataset as ds
import matplotlib.pyplot as plt


# dataset load
mtrdx, mtrdy, mtedx, mtedy = ds.load_mnist()
ftrdx, ftrdy, ftedx, ftedy = ds.load_fmnist()
fmnist_label = ['T-shirt', 'Trousers', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
ctrdx, ctrdy, ctedx, ctedy = ds.load_cifar10()
cifar_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

# plot mnist
fig_m = plt.figure('Teapot : MNIST random show')
axs = [fig_m.add_subplot(i,4,j+(i-1)*4)
            for i in range(1,3) for j in range(1,5)]

mimg = []
def onclick1(event):
    global mimg
    plt.ion()
    for i in range(len(mimg)):
        mimg[i].remove()
    mimg = []
    for i in range(len(axs)):
        if i < 4:
            pos = random.randint(0,60000)
            img = mtrdx[:,pos]
            num = np.argmax(mtrdy[:,pos])
        else:
            pos = random.randint(0,10000)
            img = mtedx[:,pos]
            num = np.argmax(mtedy[:,pos])
        tmp = axs[i].imshow(img.reshape(28,28), cmap='gray')
        mimg.append(tmp)
        axs[i].set_title(str(num))
        axs[i].axis('off')
    plt.ioff()

fig_m.subplots_adjust(top=1,wspace=0.7,hspace=0.8)
fig_m.canvas.mpl_connect('button_press_event', onclick1)

# plot fmnist
fig_f = plt.figure('Teapot : FMNIST random show')
axs2 = [fig_f.add_subplot(i,4,j+(i-1)*4)
            for i in range(1,3) for j in range(1,5)]

fimg = []
def onclick2(event):
    global fimg
    plt.ion()
    for i in range(len(fimg)):
        fimg[i].remove()
    fimg = []
    for i in range(len(axs2)):
        if i < 4:
            pos = random.randint(0,60000)
            img = ftrdx[:,pos]
            num = np.argmax(ftrdy[:,pos])
        else:
            pos = random.randint(0,10000)
            img = ftedx[:,pos]
            num = np.argmax(ftedy[:,pos])
        tmp = axs2[i].imshow(img.reshape(28,28), cmap='gray')
        fimg.append(tmp)
        axs2[i].set_title(fmnist_label[num])
        axs2[i].axis('off')
    plt.ioff()

fig_f.subplots_adjust(top=1,wspace=0.7,hspace=0.8)
fig_f.canvas.mpl_connect('button_press_event', onclick2)

# plot cifar10
fig_c = plt.figure('Teapot : CIFAR10 random show', figsize=(5,3))
axs3 = [fig_c.add_subplot(i,4,j+(i-1)*4)
            for i in range(1,3) for j in range(1,5)]


cimg = []
def onclick3(event):
    global cimg
    plt.ion()
    for i in range(len(cimg)):
        cimg[i].remove()
    cimg = []
    for i in range(len(axs3)):
        if i < 4:
            pos = random.randint(0,50000)
            img = ctrdx[:,pos]
            num = np.argmax(ctrdy[:,pos])
        else:
            pos = random.randint(0,10000)
            img = ctedx[:,pos]
            num = np.argmax(ctedy[:,pos])
        tmp = axs3[i].imshow(np.transpose(img.reshape(3,32,32),(1,2,0)))
        cimg.append(tmp)
        axs3[i].set_title(cifar_label[num])
        axs3[i].axis('off')
    plt.ioff()

fig_c.subplots_adjust(top=1,wspace=0.8,hspace=1)
fig_c.canvas.mpl_connect('button_press_event', onclick3)

# show
onclick1(None)
onclick2(None)
onclick3(None)
plt.show()


