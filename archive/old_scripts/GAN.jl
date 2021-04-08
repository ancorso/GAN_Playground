using CUDA, Flux, MLDatasets, Statistics, Images, Parameters, Printf, Random
using Base.Iterators: partition
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: train!, @epochs
using Parameters, Printf, Random

@with_kw struct HyperParameters
	batch_size::Int = 128
	latent_dim::Int = 100
	epochs::Int = 20
	verbose_freq::Int = 1000
	output_x::Int = 6
	output_y::Int = 6
	αᴰ::Float64 = 0.0002 # discriminator learning rate
	αᴳ::Float64 = 0.0002 # generator learning rate
end

function to_image(G, noise, hp)
	imgs = cpu.(G.(noise))
	img = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(imgs, hp.output_y))); dims=(3,4)), (2,1))
	img = Gray.(img .+ 1f0) ./ 2f0
	return img
end

function Discriminator()
	return Chain(
		Conv((4,4), 1=>64; stride=2, pad=1),
		x->leakyrelu.(x, 0.2f0),
		Dropout(0.25),
		Conv((4,4), 64=>128; stride=2, pad=1),
		x->leakyrelu.(x, 0.2f0),
		Dropout(0.25),
		x->reshape(x, 7*7*128, :),
		Dense(7*7*128, 1)) |>gpu
end

function Generator(hp)
	return Chain(
		Dense(hp.latent_dim, 7*7*256),
		BatchNorm(7*7*256, relu),
		x->reshape(x, 7, 7, 256, :),
		ConvTranspose((5,5), 256=>128; stride=1, pad=2),
		BatchNorm(128, relu),
		ConvTranspose((4,4), 128=>64; stride=2, pad=1),
		BatchNorm(64, relu),
		ConvTranspose((4,4), 64=>1, sigmoid; stride=2, pad=1),
		x -> 2.f0 .* x .- 1.f0) |> gpu
end


function Lᴰ(real_output, fake_output)
	real_loss = mean(logitbinarycrossentropy(real_output, 1f0, agg=identity))
	fake_loss = mean(logitbinarycrossentropy(fake_output, 0f0, agg=identity))
	return real_loss + fake_loss
end

Lᴳ(fake_output) = mean(logitbinarycrossentropy(fake_output, 1f0, agg=identity))


function train_discriminator!(G, D, noise, x, optD, hp)
	θ = Flux.params(D)
	loss, back = Flux.pullback(() -> Lᴰ(D(x), D(G(noise))), θ)
	update!(optD, θ, back(1f0))
	loss
end


function train_generator!(G, D, noise, optG, hp)
	θ = Flux.params(G)
	loss, back = Flux.pullback(() -> Lᴳ(D(G(noise))), θ)
	update!(optG, θ, back(1f0))
	loss
end


function train(;hp = HyperParameters(), G = Generator(hp), D = Discriminator(), optG = ADAM(hp.αᴳ), optD = ADAM(hp.αᴰ))
	# Load MNIST dataset
	images, _ =  MLDatasets.MNIST.traindata(Float32)
	images = reshape((2f0 .* images .- 1f0), 28, 28, 1, :) |> gpu # Normalize to [-1, 1]
	data = DataLoader(images, batchsize=hp.batch_size, shuffle = true)
	fixed_noise = [randn(hp.latent_dim, 1) |> gpu for _ in 1:hp.output_x * hp.output_y]

	# Training
	step = 0
	@epochs hp.epochs for x in data
		nD = randn(hp.latent_dim, hp.batch_size) |> gpu
		loss_D = train_discriminator!(G, D, nD, x, optD, hp)
		
		nG = randn(hp.latent_dim, hp.batch_size) |> gpu
		loss_G = train_generator!(G, D, nG, optG, hp)

		# Logging
		if step % hp.verbose_freq == 0
			@info "[$step] Discriminator loss = $loss_D, Generator loss = $loss_G"
			save(@sprintf("output/gan_%06d.png", step), to_image(G, fixed_noise, hp))
		end
		step += 1
	end
	G
end

train()

