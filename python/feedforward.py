import torch
from torch import nn, Tensor
from torch.nn.functional import relu
from memory_profiler import profile

in_dim = 4000  # 4_000
hidden_dim = 12000  # 12_000
out_dim = 221  # 221


class Residual(nn.Linear):
    def forward(self, x) -> Tensor:
        y = super().forward(x)
        return y + relu(x)


class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            Residual(hidden_dim, hidden_dim), 
            Residual(hidden_dim, hidden_dim),
            Residual(hidden_dim, hidden_dim),
            Residual(hidden_dim, hidden_dim),
            Residual(hidden_dim, hidden_dim),
            Residual(hidden_dim, hidden_dim),
            Residual(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x


@profile
def train_single(model, loss_fn, optimizer):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = torch.randn(in_dim)
y = torch.randn(out_dim)

# from timeit import timeit
# num_runs = 1000
# print(timeit(
#     "train_single(nn, loss_fn, opt)",
#     globals=globals(),
#     number=num_runs,
# ) / num_runs)
