import numpy as np

data = np.load('datafile.npz')

# Iterate over objects in the .npz filek = 0
for key in data.files:
    command=f"{key} = data[key]"
    #print(command)
    exec(command)
    
del key, command, data

