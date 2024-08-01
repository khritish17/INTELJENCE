import os
from matplotlib import pyplot as plt

path = os.path.abspath("") + "/metadata" + "/error_data.txt"
ep = []
tr_error = []
val_error = []
with open(path) as file:
    line = file.readline()
    while line:
        epoch, train_error, validation_error = line.rstrip("/n").split(",")
        epoch = int(epoch)
        train_error= float(train_error)
        validation_error = float(validation_error)
        ep.append(epoch)
        tr_error.append(train_error)
        val_error.append(validation_error)
        line = file.readline()
plt.plot(ep,tr_error)
plt.plot(ep, val_error)
plt.show()
    