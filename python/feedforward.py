import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from memory_profiler import profile

in_dim = 4000
hidden_dim = 1200
out_dim = 221


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x


network = FeedForward(in_dim, hidden_dim, out_dim)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(network.parameters())

x = torch.randn(in_dim)
y = torch.randn(out_dim)

def train_single(model, loss_fn, optimizer):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@profile
def train_single_profile(model, loss_fn, optimizer):
    train_single(model, loss_fn, optimizer)

train_single_profile(network, loss_fn, opt)

# from timeit import timeit
# num_runs = 1000
# print(timeit(
#     "train_single(network, loss_fn, opt)",
#     globals=globals(),
#     number=num_runs,
# ) / num_runs)
