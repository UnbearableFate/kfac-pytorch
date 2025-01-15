import torch
import torch.nn as nn
import torch.nn.init as init

def model2flatten_tensor(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all parameters from a given model into a single 1D tensor.

    This function extracts each parameter from the model, detaches it from the
    computation graph, and concatenates them into a single flattened tensor.

    Args:
        model (torch.nn.Module): The model whose parameters are to be flattened.

    Returns:
        torch.Tensor: A single flattened tensor containing all the model's parameters.
    """
    with torch.no_grad():
        # 使用列表推导式提取所有参数，并通过 clone() 和 detach()
        # 确保获得独立的、不追踪梯度的张量副本
        params = [p.data.detach().clone().view(-1) for p in model.parameters()]
        # 将所有参数拼接到一个张量中

        return torch.cat(params, dim=0)

def gradient2flatten_tensor(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all gradients from a given model into a single 1D tensor.

    This function extracts each parameter's gradient from the model, detaches it from the
    computation graph, and concatenates them into a single flattened tensor.

    Args:
        model (torch.nn.Module): The model whose gradients are to be flattened.

    Returns:
        torch.Tensor: A single flattened tensor containing all the model's gradients.
    """
    with torch.no_grad():
        # 使用列表推导式提取所有参数，并通过 clone() 和 detach()
        # 确保获得独立的、不追踪梯度的张量副本
        grads = [p.grad.detach().clone().view(-1) for p in model.parameters() if p.grad is not None]
        # 将所有参数拼接到一个张量中
        return torch.cat(grads, dim=0)


def flatten_tensor2model(flat: torch.Tensor, model: torch.nn.Module):
    """
    Reshape and load a flattened tensor into the model's parameters.

    Arguments:
        flat (torch.Tensor): The flattened tensor containing model parameters.
        model (torch.nn.Module): The target model to update parameters.
    """
    # Validate total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    if flat.numel() != total_params:
        raise ValueError("Flat tensor size does not match model parameters")

    # Unflatten and load parameters
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()
            param.data.copy_(flat.narrow(0, offset, numel).view_as(param))
            offset += numel

import torch

def compute_l2_norm(tensor):
    """
    Computes the L2 norm of a 1D torch.Tensor.

    Args:
        tensor (torch.Tensor): A 1D tensor.

    Returns:
        float: The L2 norm of the tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor.")
    if tensor.dim() != 1:
        raise ValueError("Input tensor must be 1-dimensional.")
    
    l2_norm = torch.sqrt(torch.sum(tensor**2))
    return l2_norm.item()  # Convert to a Python float

def get_model_total_l2_norm(model):
    """
    Compute the total L2 norm of all parameters in a PyTorch model.
    
    Args:
        model (torch.nn.Module): The model whose parameters will be evaluated.

    Returns:
        float: The total L2 norm of all parameters.
    """
    total_l2_norm = torch.sqrt(sum(torch.sum(param**2) for param in model.parameters() if param.requires_grad))
    return total_l2_norm.item()

from my_module.mobile_net import CustomMobileNetV3Small

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)  # 使用 He 初始化
        if m.bias is not None:
            init.zeros_(m.bias)

if __name__ == '__main__':
    model = CustomMobileNetV3Small()
    model.apply(initialize_weights)
    for name, param in model.named_parameters():
        print(name, param.size())
    flat = model2flatten_tensor(model)
    print(compute_l2_norm(flat))
    flat.mul_(2)
    flatten_tensor2model(flat, model)
    flat = model2flatten_tensor(model)
    print(compute_l2_norm(flat))
    for name, param in model.named_parameters():
        print(name, param.size())
