import json


class Node:
    def __init__(self, rank, world_size, io_layers):
        self.rank = rank
        self.world_size = world_size
        self.io_layers = io_layers
        self.start_target = (rank+1) % world_size

    def send_model_param(self, target, layer_name, weight, bias):
        # 模拟发送参数操作
        print(f"Node {self.rank} sends {layer_name} to Node {target}")
        data = para_store[self.rank][layer_name][-1].copy()
        old = para_store[target][layer_name][-1].copy()
        old.extend(data)
        para_store[target][layer_name].append(old)

    def send_all_model_param(self):
        target = self.start_target
        for layer_name, layer in self.io_layers.items():
            if target == self.rank:
                target = (target + 1) % self.world_size
            self.send_model_param(target, layer_name, layer.weight, layer.bias)
            target = (target + 1) % self.world_size
        self.start_target = (self.start_target + 1) % self.world_size

# 示例使用
io_layers = {
    "layer1": type('layer', (object,), {"weight": [1], "bias": [0]}),
    "layer2": type('layer', (object,), {"weight": [2], "bias": [0]}),
    "layer3": type('layer', (object,), {"weight": [3], "bias": [0]}),
    "layer4": type('layer', (object,), {"weight": [4], "bias": [0]})
}

nodes = [Node(rank, 6, io_layers) for rank in range(6)]

para_store = {}

for node in nodes:
    para_store[node.rank] = {}
    for layer_name, layer in node.io_layers.items():
        para_store[node.rank][layer_name] = [[node.rank]]

for _ in range(5):
    for node in nodes:
        node.send_all_model_param()
    print("---")

print(json.dumps(para_store, indent=4))