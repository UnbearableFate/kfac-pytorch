import torch

def fuse_tensors(tensors):
    flat_tensor = torch.cat([t.view(-1) for t in tensors])
    return flat_tensor

def fuse_model_paramenters(model):
    return fuse_tensors([p.data for p in model.parameters()])

def unfuse_tensors_to_model(flat_tensor, model):
    sizes = [param.numel() for param in model.parameters()]
    offset = 0
    for size, param in zip(sizes, model.parameters()):
        param.data.copy_(flat_tensor[offset:offset + size].view_as(param))
        offset += size