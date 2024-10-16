import torch
import os
from simulate_f32 import aot_backend
import simulate_f32 as tpu_mlir_jit
import pdb
import torch.nn as nn


import torchvision.models as models

device = torch.device("cpu")


input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

mod = models.resnet50(torch.float32)

net_d = mod
net_d.to(device)
net_d.train()
input_d = input.to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
target = torch.randint(0, 1000, (1,), dtype=torch.int64).to(device)
optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)

optimizer.zero_grad()
model_opt = torch.compile(net_d, backend=aot_backend)
for i in range(1):
    predict = model_opt(input_d)
    loss = loss_fn(predict.float(), target.long())
    loss *= 1
    loss.backward()
    optimizer.step()

