import torch
import torch.nn as nn

class ParallelLayer(nn.Module):
    def __init__(self, original_layer, split_threshold):
        super(ParallelLayer, self).__init__()
        self.split_threshold = split_threshold
        input_size = original_layer.in_features
        output_size = original_layer.out_features

        # 计算子层的数量和每个子层的输入大小
        num_sublayers = (input_size + split_threshold - 1) // split_threshold
        sublayer_input_size = (input_size + num_sublayers - 1) // num_sublayers

        self.sublayers = nn.ModuleList()
        for i in range(num_sublayers):
            start = i * sublayer_input_size
            end = min(start + sublayer_input_size, input_size)
            self.sublayers.append(nn.Linear(end - start, output_size))

        # 初始化子层的权重
        for i, sublayer in enumerate(self.sublayers):
            start = i * sublayer_input_size
            end = min(start + sublayer_input_size, input_size)
            sublayer.weight.data = original_layer.weight.data[:, start:end]

        if original_layer.bias is not None:
            self.bias = original_layer.bias
        else:
            self.bias = None

    def forward(self, x):
        sublayer_outputs = []
        input_splits = torch.split(x, [sublayer.in_features for sublayer in self.sublayers], dim=1)
        for sublayer, input_split in zip(self.sublayers, input_splits):
            sublayer_outputs.append(sublayer(input_split))
        out = sum(sublayer_outputs)
        if self.bias is not None:
            out += self.bias
        return out

class ModelSplitter(nn.Module):
    def __init__(self, origin, split_threshold=64):
        super(ModelSplitter, self).__init__()
        self.split_model = self.split_layers(origin,split_threshold)

    def split_layers(self, model,split_threshold):
        layers = []
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear) and layer.in_features > split_threshold:
                if isinstance(layers[-1],nn.Conv2d):
                    layers.append(nn.Flatten())
                layers.append(ParallelLayer(layer,split_threshold))
            elif isinstance(layer, nn.Sequential):
                layers.append(self.split_layers(layer,split_threshold))
            else:
                layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.split_model(x)