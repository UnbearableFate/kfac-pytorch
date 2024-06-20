from my_module.model_split import ModelSplitter
from my_module.custom_resnet import MLP
mlp = MLP(hidden_size=317, num_hidden_layers=5)
split_model = ModelSplitter(mlp, split_threshold=50)

for name, layer in mlp.named_children():
    print(name, layer)
print("####################")
for name, layer in split_model.named_children():
    print(name, layer)
