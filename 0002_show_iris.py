import matplotlib.pyplot as plt
import dataset as ds


cat, data = ds.get_iris()

cat0 = []
cat1 = []
cat2 = []
for it in data:
    if it[4] == '0': cat0.append(it[:4])
    if it[4] == '1': cat1.append(it[:4])
    if it[4] == '2': cat2.append(it[:4])


fig = plt.figure('Teapot: show iris', figsize=(9,5))
fig.suptitle('show iris data in 3D')
fig.subplots_adjust(wspace=0, left=0.05, right=0.95)

ax = fig.add_subplot(121, projection='3d')
for it in zip((cat0, cat1, cat2),('^','*','s')):
    x,y,z = [],[],[]
    for i in range(len(it[0])):
        x.append(float(it[0][i][0]))
        y.append(float(it[0][i][1]))
        z.append(float(it[0][i][2]))
    ax.scatter(x, y, z, marker=it[1], label=cat[0])
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')
ax.view_init(10, 20)
ax.legend()


ax2 = fig.add_subplot(122, projection='3d')
for it in zip((cat0, cat1, cat2),('^','*','s')):
    x,y,z = [],[],[]
    for i in range(len(it[0])):
        x.append(float(it[0][i][1]))
        y.append(float(it[0][i][2]))
        z.append(float(it[0][i][3]))
    ax2.scatter(x, y, z, marker=it[1], label=cat[2])
ax2.set_xlabel('sepal width')
ax2.set_ylabel('petal length')
ax2.set_zlabel('petal width')
ax2.view_init(10,-60)
ax2.legend()


plt.show()
