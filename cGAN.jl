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
	nclasses::Int = 10
	epochs::Int = 20
	verbose_freq::Int = 100
	output_x::Int = 6
	output_y::Int = 6
	αᴰ::Float64 = 0.0002 # discriminator learning rate
	αᴳ::Float64 = 0.0002 # generator learning rate
    device::Function = cpu # device to send operations to. Can be cpu or gpu
    
    # Generator architecture parameters
    latent_hidden::Int = 6272 # must be a multiple of 49
end

struct Discriminator
    d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
    d_common
end

function Discriminator(params::HyperParameters)
	d_labels = Chain(Dense(params.nclasses,784), x-> reshape(x, 28, 28, 1, size(x, 2))) |> params.device
    d_common = Chain(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2)),
                  x-> leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu),
                  x-> leakyrelu.(x, 0.2f0),
                  x-> reshape(x, :, size(x, 4)),
                  Dropout(0.4),
                  Dense(6272, 1)) |> params.device
    Discriminator(d_labels, d_common)
end

# x is the image and y is the label
function (m::Discriminator)(x, y)
    t = cat(m.d_labels(y), x, dims=3)
    return m.d_common(t)
end

struct Generator
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size)
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_common
end

function Generator(params::HyperParameters)
    g_labels = Chain(Dense(params.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x, 2))) |> params.device
    latent_channels = Int(params.latent_hidden / 49)
    g_latent = Chain(Dense(params.latent_dim, params.latent_hidden), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 7, 7, latent_channels, size(x, 2))) |> params.device
    g_common = Chain(ConvTranspose((4, 4), latent_channels+1=>latent_channels; stride=2, pad=1),
            BatchNorm(latent_channels, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), latent_channels=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64=>1, tanh; stride=1, pad=3)) |> gpu
    Generator(g_labels, g_latent, g_common)
end

# X is the noise vector and y is the desired label
function (m::Generator)(x, y)
    t = cat(m.g_labels(y), m.g_latent(x), dims=3)
    return m.g_common(t)
end


function Lᴰ(real_output, fake_output)
	real_loss = logitbinarycrossentropy(real_output, 1f0, agg=mean)
	fake_loss = logitbinarycrossentropy(fake_output, 0f0, agg=mean)
	return real_loss + fake_loss
end

Lᴳ(fake_output) = logitbinarycrossentropy(fake_output, 1f0, agg=mean)

function train_discriminator!(G, D, nx, ny, x, y, optD)
    θ = params(D.d_labels, D.d_common)
    loss, back = Flux.pullback(() -> Lᴰ(D(x, y), D(G(nx, ny), ny)), θ)
    update!(optD, θ, back(1f0))
    loss
end

function train_generator!(G, D, nx, ny, optG)
	θ = Flux.params(G.g_labels, G.g_latent, G.g_common)
	loss, back = Flux.pullback(() -> Lᴳ(D(G(nx, ny), ny)), θ)
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
   x = randn(hp.latent_dim, hp.batch_size) |> gpu
   y = Float32.(Flux.onehotbatch(rand(0:hp.nclasses-1, hp.batch_size), 0:hp.nclasses-1)) |> gpu
   x, y
end


function train(;hp = HyperParameters(), G = Generator(hp), D = Discriminator(hp), optG = ADAM(hp.αᴳ, (0.5, 0.99)), optD = ADAM(hp.αᴰ, (0.5, 0.99)))
    # Load MNIST dataset
    images, labels = MLDatasets.MNIST.traindata(Float32)
    images = reshape(2f0 .* images .- 1f0, 28, 28, 1, :) |> gpu # Normalize to [-1, 1]
    y = Float32.(Flux.onehotbatch(labels, 0:hp.nclasses-1)) |> gpu
    data = DataLoader((images, y), batchsize=hp.batch_size, shuffle = true)

    fixed_noise = [randn(hp.latent_dim, 1) |> gpu for _=1:hp.output_x * hp.output_y]
    fixed_labels = [Float32.(Flux.onehotbatch(rand(0:hp.nclasses-1, 1), 0:hp.nclasses-1)) |> gpu
                             for _ =1:hp.output_x * hp.output_y]

    # Training
    step = 0
	@epochs hp.epochs for (x, y) in data
		loss_D = train_discriminator!(G, D, rand_input(hp)..., x, y, optD)
		loss_G = train_generator!(G, D, rand_input(hp)..., optG)
        if step % hp.verbose_freq == 0
            @info("Train step $(step), Discriminator loss = $loss_D, Generator loss = $loss_G)")
            save(@sprintf("output/cgan_steps_%06d.png", step), to_image(G, fixed_noise, fixed_labels, hp))
        end
        step += 1
    end
	G
end

# Train the network:
G = train()

# Create a bunch of zeros:
fixed_noise = [randn(hp.latent_dim, 1) |> gpu for _=1:hp.output_x * hp.output_y]
fixed_labels = [Float32.(Flux.onehotbatch(rand([0], 1), 0:hp.nclasses-1)) |> gpu
						 for _ =1:hp.output_x * hp.output_y]
to_image(G, fixed_noise, fixed_labels, hp)
