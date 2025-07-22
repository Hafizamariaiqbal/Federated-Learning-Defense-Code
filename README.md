# Federated-Learning-Defense-Code

Flower + Defense Example (PyTorch, MNIST)
 
# Install requirements 

pip install flwr torch torchvision scikit-learn  
 

# 1. Define the model and client
import flwr as fl  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, Subset  
from torchvision import datasets, transforms  
import numpy as np  
from sklearn.cluster import KMeans  
  
# SimpleNet as before  
class SimpleNet(nn.Module):  
    def __init__(self):  
        super().__init__()  
        self.fc1 = nn.Linear(784, 128)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(128, 10)  
    def forward(self, x):  
        x = x.view(-1, 784)  
        x = self.relu(self.fc1(x))  
        return self.fc2(x)  
 

# 2. Flower Client
 


class FlowerClient(fl.client.NumPyClient):  
    def __init__(self, model, trainloader, testloader):  
        self.model = model  
        self.trainloader = trainloader  
        self.testloader = testloader  
  
    def get_parameters(self, config):  
        return [val.cpu().numpy() for val in self.model.state_dict().values()]  
  
    def set_parameters(self, parameters):  
        params_dict = zip(self.model.state_dict().keys(), parameters)  
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict})  
  
    def fit(self, parameters, config):  
        self.set_parameters(parameters)  
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)  
        criterion = nn.CrossEntropyLoss()  
        self.model.train()  
        for _ in range(2):  # Two local epochs for quick test  
            for data, target in self.trainloader:  
                optimizer.zero_grad()  
                output = self.model(data)  
                loss = criterion(output, target)  
                loss.backward()  
                optimizer.step()  
        return self.get_parameters({}), len(self.trainloader.dataset), {}  
  
    def evaluate(self, parameters, config):  
        self.set_parameters(parameters)  
        self.model.eval()  
        correct = 0  
        total_loss = 0  
        criterion = nn.CrossEntropyLoss()  
        with torch.no_grad():  
            for data, target in self.testloader:  
                output = self.model(data)  
                loss = criterion(output, target)  
                pred = output.argmax(dim=1)  
                correct += pred.eq(target).sum().item()  
                total_loss += loss.item() * data.size(0)  
        accuracy = correct / len(self.testloader.dataset)  
        avg_loss = total_loss / len(self.testloader.dataset)  
        return float(avg_loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}  
 

# 3. Custom Aggregation (your defense)
 

# This will be run on the server (central aggregation)  
def aggregate_with_defense(results, num_classes=10, k=2):  
    # Extract model weights  
    param_lists = [params[0] for params in results]  
    models = [torch.nn.utils.parameters_to_vector([torch.tensor(p) for p in params]) for params in param_lists]  
    models_stack = torch.stack(models)  
    # -- Defense logic below: only for demonstration on full output layer gradients --  
    # For a robust approach, you can extract the output layer part, cluster, and average only "honest" ones.  
    output_layers = models_stack[:, -1280:].numpy().reshape(len(results), num_classes, 128) # assuming last layer is fc2  
    honest_indices = set()  
    for c in range(num_classes):  
        kmeans = KMeans(n_clusters=k, random_state=0).fit(output_layers[:, c, :])  
        labels, counts = np.unique(kmeans.labels_, return_counts=True)  
        honest_label = labels[np.argmax(counts)]  
        for i, l in enumerate(kmeans.labels_):  
            if l == honest_label:  
                honest_indices.add(i)  
    if not honest_indices: honest_indices = set(range(len(results)))  
    honest_indices = list(honest_indices)  
    filtered_models = models_stack[honest_indices]  
    # Aggregate (average) honest models  
    avg_params = filtered_models.mean(dim=0)  
    # Unflatten  
    new_params = []  
    i = 0  
    for v in param_lists[0]:  
        length = np.prod(v.shape)  
        new_params.append(avg_params[i:i+length].reshape(v.shape).numpy())  
        i += length  
    return new_params  
 

# 4. Start Simulation
 

def load_data(num_clients=5):  
    transform = transforms.Compose([transforms.ToTensor()])  
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)  
    testset = datasets.MNIST('./data', train=False, download=True, transform=transform)  
    split_idx = np.array_split(np.arange(len(dataset)), num_clients)  
    trainloaders = [DataLoader(Subset(dataset, idx), batch_size=32, shuffle=True) for idx in split_idx]  
    testloader = DataLoader(testset, batch_size=128)  
    return trainloaders, testloader  
  
def client_fn(cid):  
    model = SimpleNet()  
    trainloaders, testloader = load_data()  
    return FlowerClient(model, trainloaders[int(cid)], testloader)  
  
strategy = fl.server.strategy.FedAvg(  
    fraction_fit=1.0,  
    min_fit_clients=5,  
    min_available_clients=5,  
    on_aggregate_fit=aggregate_with_defense,  # Custom defense  
)  
  
if __name__ == "__main__":  
    fl.simulation.start_simulation(  
        client_fn=client_fn,  
        num_clients=5,  
        config=fl.server.ServerConfig(num_rounds=5),  
        strategy=strategy,  
    )  
