include("taxi_models_and_data.jl")
Gconv = BSON.load("results- 09March/conv_generator_ld2.bson")[:G]
Gconv = TaxiGConv(Gconv.g_labels |> gpu, Gconv.g_latent |>gpu, Gconv.g_common |> gpu)

s = Settings(latent_dim=2, nclasses=2, G=nothing,D=nothing,loss=nothing, img_fun=nothing, rand_input=nothing, batch_size=128)
models = [
	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 16*8), x -> reshape(x, 16,8,1,:)) |> gpu),
	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 16*8), x -> reshape(x, 16,8,1,:)) |> gpu),
	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 128, relu), Dense(128, 128, relu), Dense(128, 128, relu), Dense(128, 128, relu), Dense(128, 16*8), x -> reshape(x, 16,8,1,:)) |> gpu),
	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 128, relu), Dense(128, 128, relu), Dense(128, 16*8), x -> reshape(x, 16,8,1,:)) |> gpu),
]
model_names = ["MLP256x4", "MLP256x2", "MLP128x4", "MLP128x2"]
loss = Dict()
for (m, name) in zip(models, model_names)
	θ = Flux.params(m)
	loss[name] = []
	for (bs, iter, lr) = zip([128, 256, 512, 512, 512], [10000, 5000, 1000, 1000, 1000], [1e-3, 1e-3, 1e-3, 1e-4, 1e-5])
		opt = ADAM(lr)
		println("name:", name, " bs: ", bs, " iter: ", iter )
		s = Settings(latent_dim=2, nclasses=2, G=nothing,D=nothing,loss=nothing, img_fun=nothing, rand_input=nothing, batch_size=bs)
		for i=1:iter
			x, y = Taxi_input(s)
			l, back = Flux.pullback(()-> Flux.msle(Gconv(x,y) .+ 1, m(x,y) .+ 1), θ)
			l = Flux.mse(Gconv(x,y), m(x,y))
			println("iter: $i, loss: ", l)
			push!(loss[name], l)
			update!(opt, θ, back(1f0))
		end 
	end
	G = TaxiGMLP(m.net |> cpu)
	BSON.@save "results- 09March/$(name)_msle_ld2.bson" G
end

using Plots
p = plot()
for name in model_names	
	plot!(p, loss[name], label=string(name, "-mae"), yscale=:log10)
end
plot!(p)

s = Settings(latent_dim=2, nclasses=2, G=nothing,D=nothing,loss=nothing, img_fun=nothing, rand_input=nothing, batch_size=bs)
fixed_noise = [[i, j] for i=range(-1, stop=1, length=6), j=range(-1, stop=1, length=6)][:]
fixed_labels = [zeros(2) for i=1:length(fixed_noise)]
for name in model_names
	G = BSON.load("results- 09March/$(name)_ld2.bson")[:G]
	arr = to_image(G, cpu.(fixed_noise), cpu.(fixed_labels), s)
	save("results- 09March/$(name)_mae_images.png", arr)
end 






arr = to_image(G, fixed_noise, fixed_labels, s)
save("results- 09March/smallgmlp_latentvars.png", arr)


a = [Gray.((G([dt, 0], [0; 0])[:,:,1,1]' .+ 1) ./2) for dt=-1:0.1:1]


vcat(a...)