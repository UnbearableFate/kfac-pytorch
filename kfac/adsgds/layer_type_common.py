import torch
import threading

from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
    from kfac.rpc_distributed import KFacRPCCommunicator

def rpc_work_name(rank:int) -> str:
    return f"rpc_{rank}"

class LayerwiseModelStore:
    def __init__(self, model: torch.nn.Module = None):
        self.term = 0
        self.aggration_weight = 0
        self.loss_value = 0
        self.lock = threading.Lock()
        self.layer_parameters : Dict[str, torch.Tensor]= {}
        if model is not None:
            self.register_model_content(model)
    
    def register_model_content(self, model: torch.nn.Module):
        with torch.no_grad():
            for layer_name, param in model.named_parameters():
                self.layer_parameters[layer_name] = param.data
    
    def clone_model_store(self, model_store: 'LayerwiseModelStore'):
        with self.lock:
            for layer_name, layer_parameter in model_store.layer_parameters.items():
                self.layer_parameters[layer_name] = layer_parameter.clone()
    
    def getData(self):
        return self.layer_parameters,self.term,self.loss_value
    
    def setData(self,layer_parameters,term,loss_value):
        self.layer_parameters = layer_parameters
        self.term = term
        self.loss_value = loss_value

    def setDataWithLock(self,layer_parameters,term,loss_value):
        with self.lock:
            self.layer_parameters = layer_parameters
            self.term = term
            self.loss_value = loss_value

    def aggrate_with_another_model(self, layer_tensor_param: Dict[str, torch.Tensor], recv_weight: float):
        with self.lock and torch.no_grad():
            for layer_name in self.layer_parameters.keys():
                if layer_name in layer_tensor_param:
                    self.layer_parameters[layer_name].mul_(1-recv_weight).add_(layer_tensor_param[layer_name] * recv_weight)


class RootModelAvgRPCCommunicator:
    def __init__(self, rank, model: torch.nn.Module ,rpc_communicator: 'KFacRPCCommunicator'):
        self.rpc_communicator: 'KFacRPCCommunicator' = rpc_communicator
        self.world_size_cb = rpc_communicator.get_world_size
        self.current_t_cb = self.rpc_communicator.current_t
        self.origin_world_size = rpc_communicator.origin_world_size
        self.rank = rank
        self.model = model
        self.local_model_store = LayerwiseModelStore(model)

    def set_loss(self, loss_value):
        self.local_model_store.loss_value = loss_value