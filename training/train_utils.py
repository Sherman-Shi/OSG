import collections
import numpy as np
import torch
import pdb


def to_device(x, device='cuda:0'):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		print(f'Unrecognized type in `to_device`: {type(x)}')
		pdb.set_trace()

def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def cycle(dl):
    while True:
        for data in dl:
            yield data
