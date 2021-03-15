using HDF5
using BSON
include("cGAN_common.jl")

## Taxi discriminator with convolutions
struct TaxiDConv
	d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
	d_common
end

Flux.trainable(d::TaxiDConv) = (d.d_labels, d.d_common)

function TaxiDConv(s::Settings)
	d_labels = Chain(Dense(s.nclasses,16*8), x-> reshape(x, 16, 8, 1, size(x, 2))) |> gpu
	d_common = Chain(
					 Conv((3,3), 2=>64, pad=1, stride=1),
					 BatchNorm(64, leakyrelu),
			 		 Conv((4,4), 64=>64, pad=1, stride=2),
					 BatchNorm(64, leakyrelu),
					 Conv((3,3), 64=>128, pad=1, stride=1),
					 BatchNorm(128, leakyrelu),
					 Conv((4,4), 128=>128, pad=1, stride=2),
					 BatchNorm(128, leakyrelu),
					 Conv((3,3), 128=>256, pad=1, stride=1),
					 BatchNorm(256, leakyrelu),
					 Conv((4,4), 256=>256, pad=1, stride=2),
					 BatchNorm(256, leakyrelu),
			  		 Conv((3,3), 256=>512, pad=1, stride=1),
			  		 x -> reshape(x, 1024, :),
			  		 Dense(1024, 1)
					 ) |> gpu
   TaxiDConv(d_labels, d_common)
end

function (m::TaxiDConv)(x, y)
	t = cat(m.d_labels(y), x, dims=3)
	return m.d_common(t)
end


## spectral
struct TaxiDConvSpectral
	d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
	d_common
end

Flux.trainable(d::TaxiDConvSpectral) = (d.d_labels, d.d_common)

function TaxiDConvSpectral(s::Settings)
	d_labels = Chain(DenseSN(s.nclasses,16*8), x-> reshape(x, 16, 8, 1, size(x, 2))) |> gpu
	d_common = Chain(
					 ConvSN((3,3), 2=>64, pad=1, stride=1, leakyrelu),
			 		 ConvSN((4,4), 64=>64, pad=1, stride=2, leakyrelu),
					 ConvSN((3,3), 64=>128, pad=1, stride=1, leakyrelu),
					 ConvSN((4,4), 128=>128, pad=1, stride=2, leakyrelu),
					 ConvSN((3,3), 128=>256, pad=1, stride=1, leakyrelu),
					 ConvSN((4,4), 256=>256, pad=1, stride=2, leakyrelu),
			  		 ConvSN((3,3), 256=>512, pad=1, stride=1, leakyrelu),
			  		 x -> reshape(x, 1024, :),
			  		 Dense(1024, 1)
					 ) |> gpu
   TaxiDConvSpectral(d_labels, d_common)
end

function (m::TaxiDConvSpectral)(x, y)
	t = cat(m.d_labels(y), x, dims=3)
	return m.d_common(t)
end

## Taxi discriminator with dense layers
struct TaxiDMLP
	net
end

Flux.trainable(d::TaxiDMLP) = (d.net,)


function TaxiDMLP(s::Settings)
	TaxiDMLP(Chain(Dense(16*8 + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 1, )) |> gpu)
end

function (m::TaxiDMLP)(x, y)
	m.net(vcat(reshape(x, :, size(y,2)), y))
end
	

## Taxi generator with conv layers
struct TaxiGConv
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size) 
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_common    
end

Flux.trainable(d::TaxiGConv) = (d.g_labels, d.g_latent, d.g_common)

function TaxiGConv(s::Settings)
	g_labels = Chain(Dense(s.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x, 2))) |> gpu
    g_latent = Chain(Dense(s.latent_dim, 6272), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 7, 7, 128, size(x, 2))) |> gpu
    g_common = Chain(ConvTranspose((4, 4), 129=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((5, 5), 64=>1, tanh; stride=(2, 4), pad=4)) |> gpu
	TaxiGConv(g_labels, g_latent, g_common)
end

function (m::TaxiGConv)(x, y)
    t = cat(m.g_labels(y), m.g_latent(x), dims=3)
    return m.g_common(t)
end

## Taxi generator with mlp
struct TaxiGMLP
	net
end

Flux.trainable(d::TaxiGMLP) = (d.net,)

function BigMLP(s::Settings)
 	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 16*8), x -> reshape(x, 16,8,1,:)) |> gpu)
end

function SmallMLP(s::Settings)
 	TaxiGMLP(Chain(Dense(s.latent_dim + s.nclasses, 128, relu), Dense(128, 128, relu), Dense(128, 16*8), x -> reshape(x, 16,8,1,:)) |> gpu)
end
 
function (m::TaxiGMLP)(x, y)
	m.net(vcat(reshape(x, :, size(y,2)), y))
end

## Functions for generating outputs

# Function that returns random input for the generator
function Taxi_input(s::Settings)
   # x = randn(s.latent_dim, s.batch_size) |> gpu
   x = Float32.(rand(Uniform(-1.0, 1.0), s.latent_dim, s.batch_size)) |> gpu
   y = Float32.(vcat(rand(Uniform(-1.73, 1.73), 1, s.batch_size), 
   					 rand(Uniform(-1.74, 1.74), 1, s.batch_size), 
					 # rand(Uniform(-1.7, 1.7), 1, s.batch_size)
					 )) |> gpu
   # y = zeros(Float32, 2, s.batch_size) |> gpu
   x, y
end

function gen_Taxi_images(s::Settings)
	# Load the data
	# fn = "KJ_DownsampledTrainingData.h5"
	# images = h5read(fn, "X_train")
	# y = Float32.(h5read(fn, "y_train"))
	
	fn = "data/SK_DownsampledGANFocusAreaData.h5"
	images = h5read(fn, "y_train") # yes I know the labels seem backwards
	y = h5read(fn, "X_train")
	println("std1: ", std(y[1,:]), " std2: ", std(y[2,:]))
	y[1,:] ./= std(y[1,:])
	y[2,:] ./= std(y[2,:])
	
	down_start = y[3,1]
	dash_distance = 30.45 #200/6.5
	y[3,:] .= rem.((y[3,:] .- down_start), dash_distance)
	y[3,:] .= (y[3,:] .- mean(y[3,:]))./std(y[3,:])
	println("extrema of dim1: ", extrema(y[1,:]), " extrema of dim2: ", extrema(y[2,:]), "extrema of dim3: ", extrema(y[3,:]))
	y = y |> gpu

	# Generate fixed set of labels and save the corresponding images
	N = s.output_x * s.output_y
	fixed_noise = [rand(Uniform(-1.0, 1.0), s.latent_dim, 1) |>gpu for _=1:N]
	
	indices = rand(MersenneTwister(0), 1:size(y,2), N)
	# fixed_labels = [[zeros(2)..., i*2/N - 1] |> gpu for i=1:N]
	# fixed_labels = [zeros(2) |> gpu for i=1:N]
	fixed_labels = [y[1:2, i] for i in indices]
	real_images = [cpu(images[:,:, i]) for i in indices]
	image_array = permutedims(reduce(vcat, reduce.(hcat, partition(real_images, s.output_y))), (2,1))
	image_array = Gray.(image_array)
	save("real_images_sk.png", image_array)

	# rescale the data
	images = reshape(2f0 .* images .- 1f0, 16, 8, 1, :) |> gpu # Normalize to [-1, 1]
	data = DataLoader((images, y[1:2,:]), batchsize=s.batch_size, shuffle = true, partial = false)
								 
	data, fixed_noise, fixed_labels
end

# First reduce the latent dimension size
# epochs = 500
# 
# ld = 2
# G, D = train(Settings(G = TaxiGConv, D = TaxiDConvSpectral, epochs = epochs, rand_input = Taxi_input, loss = DCGANLoss(), img_fun = gen_Taxi_images, nclasses=2, latent_dim = ld, output_dir = "no_downtrack_DCGAN_ld$(ld)_dt"))
# 
# G = TaxiGConv(G.g_labels |> gpu, G.g_latent |> gpu, G.g_common |> gpu)
# BSON.@save "generator_ld$(ld).bson" G
