using CUDA, Flux, MLDatasets, Statistics, Images, Parameters, Printf, Random
using Base.Iterators: partition
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Flux.Data: DataLoader
using Flux.Optimise: train!, @epochs
using Parameters, Printf, Random
using BSON: @save
using Distributions

CUDA.allowscalar(false)

using StaticArrays
include("../SimpleRender/simple_render.jl")


@with_kw struct HyperParameters
	batch_size::Int = 128
	# latent_dim::Int = 100
	nclasses::Int = 2
	epochs::Int = 120
	verbose_freq::Int = 1000
	output_x::Int = 6
	output_y::Int = 6
	αᴰ::Float64 = 0.0002 # discriminator learning rate
	αᴳ::Float64 = 0.0002 # generator learning rate
end

struct Discriminator
    d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
    d_common   
end

function Discriminator(hp::HyperParameters)
	d_labels = Chain(Dense(hp.nclasses, 14*14), x-> reshape(x, 14,14, 1, size(x, 2))) |> gpu
    d_common = Chain(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2)),
                  x-> leakyrelu.(x, 0.2f0),
                  Dropout(0.4),
                  Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu),
                  x-> leakyrelu.(x, 0.2f0),
                  x-> reshape(x, :, size(x, 4)),
                  Dropout(0.4),
                  Dense(2048, 1)) |> gpu
    Discriminator(d_labels, d_common)
end

# x is the image and y is the label
function (m::Discriminator)(x, y)
    t = cat(m.d_labels(y), x, dims=3)
    return m.d_common(t)
end

struct Generator
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size) 
    g_common    
end


function Generator(hp::HyperParameters)
	g_labels = Chain(Dense(hp.nclasses, 49*7), x-> reshape(x, 7 , 7 , 7 , size(x, 2))) |> gpu
    g_common = Chain(ConvTranspose((4, 4), 7=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64=>1; stride=1, pad=4),
			Conv((3, 3), 1=>1, tanh; stride=2)) |> gpu
    Generator(g_labels, g_common)
end

# X is the noise vector and y is the desired label
function (m::Generator)(y)
    return m.g_common(m.g_labels(y))
end


function Lᴰ(real_output, fake_output)
	real_loss = logitbinarycrossentropy(real_output, 1f0, agg=mean)
	fake_loss = logitbinarycrossentropy(fake_output, 0f0, agg=mean)
	return real_loss + fake_loss
end

Lᴳ(fake_output) = logitbinarycrossentropy(fake_output, 1f0, agg=mean)

function train_discriminator!(G, D, ny, x, y, optD)
    θ = Flux.params(D.d_labels, D.d_common)
    loss, back = Flux.pullback(() -> Lᴰ(D(x, y), D(G(ny), ny)), θ)
    update!(optD, θ, back(1f0))
    loss
end

function train_generator!(G, D, ny, optG)
	θ = Flux.params(G.g_labels, G.g_common)
	loss, back = Flux.pullback(() -> Lᴳ(D(G(ny), ny)), θ)
	update!(optG, θ, back(1f0))
	loss
end

function to_image(G, fixed_state, hp)
    fake_images = cpu.(G.(fixed_state))
    image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(fake_images, hp.output_y))); dims=(3, 4)), (2, 1))
    image_array = Gray.(image_array .+ 1f0) ./ 2f0
    return image_array
end

rand_state(batch_size) = Float32.(vcat(rand(Uniform(deg2rad(-22), deg2rad(22)), 1, batch_size), rand(Uniform(-3, 3),1,  batch_size)))

function train(; savedir = "output", hp = HyperParameters(), G = Generator(hp), D = Discriminator(hp), optG = ADAM(hp.αᴳ, (0.5, 0.99)), optD = ADAM(hp.αᴰ, (0.5, 0.99)))
	load_data = BSON.load("pendulum_data.bson")
	images, states = load_data[:images] |> gpu, load_data[:states] |> gpu
	data = DataLoader((images, states), batchsize=hp.batch_size, shuffle = true)
	
	fixed_states = [rand_state(1) |> gpu for i=1:hp.output_x*hp.output_y]
	
    # Training
	step = 0
	@epochs hp.epochs for (x, y) in data
		loss_D = train_discriminator!(G, D, rand_state(hp.batch_size) |> gpu, x, y, optD)
		loss_G = train_generator!(G, D, rand_state(hp.batch_size) |> gpu, optG)

        if step % hp.verbose_freq == 0
            @info("Train step $(step), Discriminator loss = $loss_D, Generator loss = $loss_G)")
			name = @sprintf("cgan_steps_%06d.png", step)
            save(string(savedir, "/", name), to_image(G, fixed_states, hp))
        end
        step += 1
    end
	G, fixed_states
end  

G, fixed_states = train()


hp = HyperParameters()
real_images = reshape.(simple_render_pendulum.(cpu(fixed_states)), 14, 14, 1, 1)
image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(real_images, hp.output_y))); dims=(3, 4)), (2, 1))
image_array = Gray.(image_array)
save("real_images.png", image_array)

# # Generate training data
# Nsamples = 10000
# states = rand_state(Nsamples)
# images = zeros(Float32, 14, 14, Nsamples)
# for i=1:Nsamples
# 	images[:,:,i] .= simple_render_pendulum(states[:, i])
# end
# images = reshape(2f0 .* images .- 1f0, 14, 14, 1, :)
# 
# @save "pendulum_data.bson" states images
