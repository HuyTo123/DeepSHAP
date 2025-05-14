import torch
import numpy as np
arr = []
tensor1= torch.rand(3, 2)
tensor2 =  torch.ones(3,2)
tensor3 = torch.where(
    torch.abs(tensor1) < 0.5,
    tensor2,
    tensor1
)
print(tensor3)
