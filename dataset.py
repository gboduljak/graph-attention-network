from torch_geometric.loader import DataLoader

num_features = 50
num_labels = 121

train_loader: DataLoader = None
val_loader: DataLoader = None
test_loader: DataLoader = None