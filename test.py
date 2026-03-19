# %%
print("Hello")

# %%
print("Hello from Jupyter in VS Code!")

print("Jupyter in VS Code!")

print("Jupyter in VS Code!")

# %%


import sys
print(sys.executable)

# %%


# %%
#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D

# %%
#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
#nn.Module
class (nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return
# %%
#dataset
class MyDataset(D.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        return

    def __len__(self):
        return
# %%
#figax
fig, ax = plt.subplots()
# %%
#dcn
.detach().cpu().numpy()
# %%
#td
.to(device)

# %%
'''
02.26.2026 Test codes
'''
import numpy as np
data = np.arange(3*3).reshape(3,3)
data.shape
np.max(data)
np.max(data, axis=1)
np.max(data, axis=0)
np.sum(data, axis=1)

def f():
    return 1,2

a, b = f()
print(a, b)

_
i
data

outputs
outputs.shape
torch.max(outputs, 1)
torch.argmax(outputs, 1)

len(test_loader) % 64

data_i=2
plt.matshow(images[data_i].numpy().squeeze(0))
outputs[data_i]