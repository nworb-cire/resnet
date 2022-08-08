using PyCall, BenchmarkTools
@pyinclude("feedforward.py")

network = py"FeedForward(in_dim, hidden_dim, out_dim)"
loss_fn = py"torch.nn.MSELoss()"
opt = py"torch.optim.Adam($network.parameters())"

t = @benchmark network($(py"x"))
display(t)
t = @benchmark py"train_single"(network, loss_fn, opt)
display(t)
