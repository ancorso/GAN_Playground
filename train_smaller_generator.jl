include("taxi_models_and_data.jl")
ld = 2
Gconv = BSON.load("conv_generator_results/BCE_BS256_LR0.0007/conv_generator_ld$(ld).bson")[:G]
Gconv = TaxiGConv(Gconv.g_labels |> gpu, Gconv.g_latent |>gpu, Gconv.g_common |> gpu)

D = BSON.load("conv_generator_results/BCE_BS256_LR0.0007/conv_discriminator_ld$(ld).bson")[:D]
D = TaxiDConvSpectral(D.d_labels |> gpu, D.d_common |> gpu)

## Parameters
batch_size = 256
iter = 50000
verbose_freq = 1000

for λ in [0f0, 1f-4, 1f-3, 1f-2, 1f-1, 1f0]
	for lr in [7f-4, 7f-3]
		m = TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu, init=Flux.orthogonal), Dense(256, 256, relu, init=Flux.orthogonal), Dense(256, 256, relu, init=Flux.orthogonal), Dense(256, 256, relu, init=Flux.orthogonal), Dense(256, 16*8, init=Flux.orthogonal), x -> reshape(x, 16,8,1,:)) |> gpu)
		output_dir = "mlp_generator_results/MLP256x4_mae_adv_ld$(ld)_λ$(λ)_LR$(lr)"

		s = Settings(latent_dim=ld, nclasses=2, G=nothing,D=nothing,loss=nothing, img_fun=nothing, rand_input=nothing, batch_size=batch_size)
		data, fixed_noise, fixed_labels = gen_Taxi_images(s)
		θ = Flux.params(m)
		opt = ADAM(lr)
		losses = []
		for i=0:iter
			x, y = Taxi_input(s)
			l, back = Flux.pullback(()-> begin
				xtilde = m(x,y)
				Flux.mae(Gconv(x,y), xtilde) - λ*mean(tanh.(D(xtilde, y))) + orthogonal_regularization(m)
			end, θ)
			if i % verbose_freq == 0
				name = @sprintf("mlpgan_iter_%06d.png", i)
				save(string(output_dir, "/", name), to_image(m, fixed_noise, fixed_labels, s))
				println("iter: $i, loss: ", l)
			end
			push!(losses, l)
			update!(opt, θ, back(1f0))
		end 
		G = TaxiGMLP(m.net |> cpu)
		BSON.@save "$(output_dir)/mlp_generator.bson" G
		BSON.@save "$(output_dir)/losses.bson" losses
	end
end




output_dir = "BCE_MLP256x4_pretrain_BS256_LR0.0007"
Gpretrain = BSON.load("MLP256x4_mae_ld2.bson")[:G]
Gpretrain = TaxiGMLP(Gpretrain.net |> gpu)





to_image(Gpretrain, fixed_noise, fixed_labels)
G, D, Ghist, Dhist = train(Settings(G=(args...)-> Gpretrain, 
								  D=(args...)->D, 
								  epochs=500, 
								  batch_size=256, 
								  rand_input=Taxi_input, 
								  loss=LSLoss(), 
								  img_fun=gen_Taxi_images, 
								  nclasses=2, 
								  latent_dim=ld,
								  verbose_freq=10,
								  optD = ADAM(7f-4, (0.5, 0.99)),
								  optG = ADAM(7f-4, (0.5, 0.99)),
								  output_dir = "$(output_dir)",))

using Plots
p = plot()
for name in model_names	
	plot!(p, loss[name], label="$(name)_mse_ld$(ld)", yscale=:log10)
end
plot!(p)
savefig("conv_loss_curves.png")


s = Settings(latent_dim=ld, nclasses=2, G=nothing,D=nothing,loss=nothing, img_fun=nothing, rand_input=nothing, batch_size=128)
fixed_noise = [rand(Uniform(-1,1), ld, 1) for i=1:36]
fixed_labels = [zeros(2) for i=1:length(fixed_noise)]
for name in model_names
	G = BSON.load("shrinking_generator_results/three_ld/$(name)_mae_ld$(ld).bson")[:G]
	arr = to_image(G, cpu.(fixed_noise), cpu.(fixed_labels), s)
	save("shrinking_generator_results/three_ld/$(name)_mae_ld$(ld)images.png", arr)
end 

G = BSON.load("conv_generator_ld3.bson")[:G]
arr = to_image(G, cpu.(fixed_noise), cpu.(fixed_labels), s)
save("Gconv_3ld_images.png", arr)

