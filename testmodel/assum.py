import torch
import numpy as np
arr = []
tensor1= torch.rand(10, 3, 32, 32)
tesnor2 = torch.rand(10, 3, 32, 32)
arr.append(tensor1)
arr.append(tesnor2)

print(arr[0][1].shape)
for j  in range(2):
    
    print(arr[0][j : j +1].shape)
