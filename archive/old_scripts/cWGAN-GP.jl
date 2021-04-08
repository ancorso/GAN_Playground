using CUDA, Flux, MLDatasets, Statistics, Images, Parameters, Printf, Random
using Base.Iterators: partition
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: train!, @epochs
using Parameters, Printf, Random
using BSON: @save
using Distributions

@with_kw struct HyperParameters
	batch_size::Int = 128
	latent_dim::Int = 100
	nclasses::Int = 10
	epochs::Int = 200
	verbose_freq::Int = 1000
	output_x::Int = 6
	output_y::Int = 6
	αᴰ::Float64 = 0.0002f0 # discriminator learning rate
	αᴳ::Float64 = 0.0002f0 # generator learning rate
end

struct Discriminator
    d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
    d_common   
end

function Discriminator(params::HyperParameters)
	# d_common = Chain(Dense(784 + params.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 1, )) |> gpu
	d_labels = Chain(Dense(params.nclasses,784), x-> reshape(x, 28, 28, 1, size(x, 2))) |> gpu
    d_common = Chain(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2)),
                  x-> leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu),
                  x-> leakyrelu.(x, 0.2f0),
                  x-> reshape(x, :, size(x, 4)),
                  Dropout(0.4),
                  Dense(6272, 1)) |> gpu
    Discriminator(d_labels, d_common)
end

# x is the image and y is the label
function (m::Discriminator)(x, y)
    t = cat(m.d_labels(y), x, dims=3)
    return m.d_common(t) 
	# return m.d_common(vcat(reshape(x, :, size(y,2)), y))
end

struct Generator
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size) 
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_common    
end


function Generator(params::HyperParameters)
	g_labels = Chain(Dense(params.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x, 2))) |> gpu
    g_latent = Chain(Dense(params.latent_dim, 6272), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 7, 7, 128, size(x, 2))) |> gpu
    g_common = Chain(ConvTranspose((4, 4), 129=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64=>1, tanh; stride=1, pad=3)) |> gpu
	# g_common = Chain(Dense(params.latent_dim + params.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 784, tanh), x -> reshape(x, 28,28,1,:)) |> gpu
    Generator(g_labels, g_latent, g_common)
end

# X is the noise vector and y is the desired label
function (m::Generator)(x, y)
    t = cat(m.g_labels(y), m.g_latent(x), dims=3)
    return m.g_common(t)
	# return m.g_common(vcat(reshape(x, :, size(y,2)), y))
end

function gradient_penalty(D, x, y)
	B = size(y, 2)
	l, b = Flux.pullback(() -> D(x, y), Flux.params(x, y))
	grads = b(ones(Float32, 1, size(x, 4)) |> gpu)
	Flux.mean((sqrt.(sum(reshape(grads[x], :, B).^2, dims = 1) .+ sum(grads[y].^2, dims = 1)) .- 1f0).^2)
end

function approx_penalty(D, x, xhat, y, yhat)
	B = size(x, 4) # batch size
	Δx, Δy = reshape(x .- xhat, :, B), y .- yhat # difference
	mag = sqrt.(sum(Δx.^2, dims = 1) + sum(Δy.^2, dims = 1))
	xdir, ydir = reshape(Δx ./ mag, size(xhat)), Δy ./ mag
	δ = 0.1f0
	ΔD = abs.(D(xhat, yhat) .- D(xhat .+ δ .* xdir, yhat .+ δ .* ydir))
	# Flux.mean((ΔD ./ δ .- 1f0).^2)
	Flux.mean(max.((ΔD ./ δ .- 1f0), 0f0).^2)
end

function Lᴰ(G, D, z, ny, x, y, ϵ, λ)
	xtilde = G(z, ny)
	ϵx = reshape(ϵ, 1, 1, 1, length(ϵ))
	xhat = ϵx .* xtilde + (1f0 .- ϵx) .* x
	yhat = ϵ .* ny + (1f0 .- ϵ) .* y
	mean(D(xtilde, ny) .- D(x, y)) + λ*approx_penalty(D, x, xhat, y, yhat)#+ λ*gradient_penalty(D, xhat, yhat)
end

Lᴳ(G, D, z, ny) = -mean(D(G(z, ny), ny)) 


function train_discriminator!(G, D, z, ny, x, y, ϵ, λ, optD)
    θ = Flux.params(D.d_labels, D.d_common)
    loss, back = Flux.pullback(() -> Lᴰ(G, D, z, ny, x, y, ϵ, λ), θ)
	loss isa Float64 && error("Loss is double precision")
    update!(optD, θ, back(1f0))
    loss
end

function train_generator!(G, D, z, ny, optG)
	θ = Flux.params(G.g_labels, G.g_latent, G.g_common)
	loss, back = Flux.pullback(() -> Lᴳ(G, D, z, ny), θ)
	loss isa Float64 && error("Loss is double precision")
	update!(optG, θ, back(1f0))
	loss
end

function to_image(G, fixed_noise, fixed_labels, hp)
    fake_images = cpu.(G.(fixed_noise, fixed_labels))
    image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hp.output_y))); dims=(3, 4)), (2, 1))
    image_array = Gray.(image_array .+ 1f0) ./ 2f0
    return image_array
end
 
# Function that returns random input for the generator
function rand_input(hp)
   z = randn(hp.latent_dim, hp.batch_size) |> gpu
   y = Float32.(Flux.onehotbatch(rand(0:hp.nclasses-1, hp.batch_size), 0:hp.nclasses-1)) |> gpu
   z, y
end

function MNIST_images(hp)
	# Load MNIST dataset
	images, labels = MLDatasets.MNIST.traindata(Float32)
	images = reshape(2f0 .* images .- 1f0, 28, 28, 1, :) |> gpu # Normalize to [-1, 1]
	y = Float32.(Flux.onehotbatch(labels, 0:hp.nclasses-1)) |> gpu
	data = DataLoader((images, y), batchsize=hp.batch_size, shuffle = true, partial = false)

	fixed_noise = [randn(hp.latent_dim, 1) |> gpu for _=1:hp.output_x * hp.output_y]
	fixed_labels = [Float32.(Flux.onehotbatch(rand(0:hp.nclasses-1, 1), 0:hp.nclasses-1)) |> gpu 
							 for _ =1:hp.output_x * hp.output_y]
								 
	data, fixed_noise, fixed_labels
end


function train(image_fn; 
				savedir = "cWGAN-GP", 
				hp = HyperParameters(), 
				G = Generator(hp), 
				D = Discriminator(hp), 
				optG = #=RMSProp(hp.αᴳ),=# ADAM(hp.αᴳ, (0.5, 0.99)),
				optD = #=RMSProp(hp.αᴰ),=# ADAM(hp.αᴰ, (0.5, 0.99)),
				λ = 10f0,
				n_disc_steps = 5,
				)
	data, fixed_noise, fixed_labels = image_fn(hp)
	
    # Training
	step = 0
	loss_G = 0
	@epochs hp.epochs for (x, y) in data
		ϵ = Float32.(rand(Uniform(0, 1), 1, size(y,2))) |> gpu
		loss_D = train_discriminator!(G, D, rand_input(hp)..., x, y, ϵ, λ, optD)
		if step % n_disc_steps == 0
			loss_G = train_generator!(G, D, rand_input(hp)..., optG)
		end
		
        if step % hp.verbose_freq == 0
            @info("Train step $(step), Discriminator loss = $loss_D, Generator loss = $loss_G)")
			name = @sprintf("cgan_steps_%06d.png", step)
            save(string(savedir, "/", name), to_image(G, fixed_noise, fixed_labels, hp))
        end
        step += 1
    end
	G
end  

# Train the network:
G = train(MNIST_images)

