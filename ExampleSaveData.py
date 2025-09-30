
import numpy as np


array1 = np.array([1, 2, 3, 4, 5])
data2 = np.array([1, 2, 3])


# Save data as .npz
filename = f"./datafile.npz"  # f-string 

np.savez(filename,
           array1=array1,
           data2=data2)

# globals().clear()