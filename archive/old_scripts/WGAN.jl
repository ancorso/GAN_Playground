using CUDA, Flux, MLDatasets, Statistics, Images, Parameters, Printf, Random
using Base.Iterators: partition
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: train!, @epochs
using Parameters, Printf, Random, Distributions

@with_kw struct HyperParameters
	batch_size::Int = 128
	latent_dim::Int = 100
	epochs::Int = 100
	verbose_freq::Int = 1000
	output_x::Int = 6
	output_y::Int = 6
	αᴰ::Float64 = 0.001 # discriminator learning rate
	αᴳ::Float64 = 0.001 # generator learning rate
end

function to_image(G, noise, hp)
	imgs = cpu.(G.(noise))
	img = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(imgs, hp.output_y))); dims=(3,4)), (2,1))
	img = Gray.(img .+ 1f0) ./ 2f0
	return img
end

function Discriminator()
	fc_disc = Chain(Dense(7*7*128, 1024), BatchNorm(1024), 
						 x->leakyrelu.(x, 0.2f0), Dense(1024, 1))
	conv_ = Chain(Conv((4,4), 1=>64;stride=(2,2), pad=(1,1)), x->leakyrelu.(x, 0.2f0),
	             Conv((4,4), 64=>128; stride=(2,2), pad=(1,1)), BatchNorm(128), 
				 x->leakyrelu.(x, 0.2f0))
	
	discriminator = Chain(conv_..., x->reshape(x, 7*7*128, :), fc_disc...) |> gpu
	

################################## Discriminator ###############################


	# return Chain(x -> reshape(x, 784, :), Dense(784, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 1, )) |> gpu
	
		# Conv((4,4), 1=>64; stride=2, pad=1),
		# x->leakyrelu.(x, 0.2f0),
		# Dropout(0.25),
		# Conv((4,4), 64=>128; stride=2, pad=1),
		# x->leakyrelu.(x, 0.2f0),
		# Dropout(0.25),
		# x->reshape(x, 7*7*128, :),
		# Dense(7*7*128, 1)) |>gpu
end

function Generator(hp)
	fc_gen = Chain(Dense(hp.latent_dim, 1024), BatchNorm(1024, relu),
            Dense(1024, 7*7*128), BatchNorm(7*7*128, relu))
	deconv_ = Chain(ConvTranspose((4,4), 128=>64; stride=(2,2),pad=(1,1)), BatchNorm(64, relu),
	                ConvTranspose((4,4), 64=>1, sigmoid; stride=(2,2), pad=(1,1)), x -> 2.f0 .* x .- 1.f0)
	
	generator = Chain(fc_gen..., x -> reshape(x, 7, 7, 128, :), deconv_...) |> gpu
	# return Chain(Dense(hp.latent_dim, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 784, tanh), (x)->reshape(x, 28,28,1,:)) |> gpu
		
		# BatchNorm(7*7*256, relu),
		# x->reshape(x, 7, 7, 256, :),
		# ConvTranspose((5,5), 256=>128; stride=1, pad=2),
		# BatchNorm(128, relu),
		# ConvTranspose((4,4), 128=>64; stride=2, pad=1),
		# BatchNorm(64, relu),
		# ConvTranspose((4,4), 64=>1, sigmoid; stride=2, pad=1),
		# x -> 2.f0 .* x .- 1.f0) |> gpu
end

function gradient_penalty(D, x)
	l, b = Flux.pullback(() -> D(x), Flux.params(x))
	grads = b(ones(Float32, 1, size(x, 4)) |> gpu)
	Flux.mean((sqrt.(sum(grads[x].^2, dims = 1)) .- 1f0).^2)
end

function approx_penalty(D, x, xhat)
	Δx = x .- xhat # difference
	B = size(x, 4) # batch size
	xnorm = sqrt.(sum(Δx.^2, dims = (1,2,3)))
	dir = Δx ./ xnorm
	δ = 0.1f0
	ΔD = abs.(D(xhat) .- D(xhat .+ δ .* dir))
	# Flux.mean((ΔD ./ δ .- 1f0).^2)
	Flux.mean(max.((ΔD ./ δ .- 1f0), 0f0).^2)
end


function Lᴰ(G, D, noise, x, λ, ϵ)
	xtilde = G(noise)
	ϵx = reshape(ϵ, 1, 1, 1, length(ϵ))
	xhat = ϵx .* xtilde + (1f0 .- ϵx) .* x
	return mean(D(xtilde) .- D(x)) + λ*approx_penalty(D, x, xhat)
end

Lᴳ(G, D, noise) = -mean(D(G(noise)))


function train_discriminator!(G, D, noise, x, λ, ϵ, optD, hp)
	θ = Flux.params(D)
	loss, back = Flux.pullback(() -> Lᴰ(G, D, noise, x,  λ, ϵ), θ)
	update!(optD, θ, back(1f0))
	# for p in θ
	# 	clamp!(p, -0.01f0, 0.01f0)
	# end
	loss
end


function train_generator!(G, D, noise, optG, hp)
	θ = Flux.params(G)
	loss, back = Flux.pullback(() -> Lᴳ(G, D, noise), θ)
	update!(optG, θ, back(1f0))
	loss
end


function train(;hp = HyperParameters(), G = Generator(hp), D = Discriminator(), λ = 10f0,  optG = RMSProp(hp.αᴳ), optD = RMSProp(hp.αᴰ))
	# Load MNIST dataset
	images, _ =  MLDatasets.MNIST.traindata(Float32)
	images = reshape((2f0 .* images .- 1f0), 28, 28, 1, :) |> gpu # Normalize to [-1, 1]
	data = DataLoader(images, batchsize=hp.batch_size, shuffle = true, partial = false)
	fixed_noise = [randn(hp.latent_dim, 1) |> gpu for _ in 1:hp.output_x * hp.output_y]

	# Training
	step = 0
	loss_G = 0
	@epochs hp.epochs for x in data
		nD = randn(hp.latent_dim, hp.batch_size) |> gpu
		ϵ = Float32.(rand(Uniform(0, 1), 1,hp.batch_size)) |> gpu
		loss_D = train_discriminator!(G, D, nD, x, λ, ϵ, optD, hp)
		if step % 5 == 0
			nG = randn(hp.latent_dim, hp.batch_size) |> gpu
			loss_G = train_generator!(G, D, nG, optG, hp)
		end

		# Logging
		if step % hp.verbose_freq == 0
			@info "[$step] Discriminator loss = $loss_D, Generator loss = $loss_G"
			save(@sprintf("WGAN-GP/gan_%06d.png", step), to_image(G, fixed_noise, hp))
		end
		step += 1
	end
	G
end

train()

