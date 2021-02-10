using Flux

include("spectral_norm.jl")

x = range(-π, stop=π, length=10)
y = sin.(x)'

p = [1,2,3]'
xx = x .^ p

m = DenseSN(3, 1, n_iterations = 100)
m.W .= [-0.42586392 -0.01890341  0.325393]
m.b .= 0.4772896

m(xx')
loss = Flux.mse(m(xx'), y)

g = gradient(() -> Flux.mse(m(xx'), y), Flux.params(m))

g.grads[m.W]
g.grads[m.b]


# grad should be: tensor([[234.2348,  11.5681, 307.2310]])

m.layer.W

