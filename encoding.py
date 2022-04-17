import numpy as np
import torch

def frequency_encode(x, L, include_input=True):
    # x: [B, ..., k]
    x_encoded = []
    for l in range(L):
        x_encoded.append(torch.sin((2**l)*np.pi*x))
        x_encoded.append(torch.cos((2**l)*np.pi*x))
    x_encoded = torch.cat(x_encoded, dim=-1)  # x_encoded: [B, 2*L*K]
    if include_input:
        x_encoded = torch.cat([x_encoded, x], dim=-1)
    return x_encoded