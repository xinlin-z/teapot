import random
import numpy as np
import dataset as ds
import matplotlib.pyplot as plt


ctrdx, ctrdy, ctedx, ctedy = ds.load_cifar10()
cifar_label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']

# plot cifar10 
fig_c = plt.figure('Teapot : CIFAR10 random show', figsize=(5,3))
axs3 = [fig_c.add_subplot(i,4,j+(i-1)*4) 
            for i in range(1,3) for j in range(1,5)]


cimg = []
def onclick(event):
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
fig_c.canvas.mpl_connect('button_press_event', onclick)

onclick(None)
plt.show()

