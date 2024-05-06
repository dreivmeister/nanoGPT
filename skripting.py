import torch

a = torch.randn((12, 64, 128))

print(a[:,1,:].size())

fc = torch.nn.Linear(128, 4*128)

print(fc(a).size())