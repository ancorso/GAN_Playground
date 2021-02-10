include("cGAN_common.jl")

## MNIST discriminator with convolutions
struct MNISTDConv
	d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
	d_common
end

Flux.trainable(d::MNISTDConv) = (d.d_labels, d.d_common)

function MNISTDConv(s::Settings)
	d_labels = Chain(Dense(s.nclasses,784), x-> reshape(x, 28, 28, 1, size(x, 2))) |> gpu
	d_common = Chain(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2)),
				  x-> leakyrelu.(x, 0.2f0),
				  Dropout(0.4),
				  Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu),
				  x-> leakyrelu.(x, 0.2f0),
				  x-> reshape(x, :, size(x, 4)),
				  Dropout(0.4),
				  Dense(6272, 1)) |> gpu
   MNISTDConv(d_labels, d_common)
end

function (m::MNISTDConv)(x, y)
	t = cat(m.d_labels(y), x, dims=3)
	return m.d_common(t)
end


## Spectral norm d Conv
struct MNISTDConvSpectral
	d_labels		# Submodel to take labels as input and convert them to the shape of image ie. (28, 28, 1, batch_size)
	d_common
end

Flux.trainable(d::MNISTDConvSpectral) = (d.d_labels, d.d_common)

function MNISTDConvSpectral(s::Settings)
	d_labels = Chain(SpectralNorm(Dense(s.nclasses,784)), x-> reshape(x, 28, 28, 1, size(x, 2))) |> gpu
	d_common = Chain(SpectralNorm(Conv((3,3), 2=>128, pad=(1,1), stride=(2,2))),
				  x-> leakyrelu.(x, 0.2f0),
				  Dropout(0.4),
				  SpectralNorm(Conv((3,3), 128=>128, pad=(1,1), stride=(2,2), leakyrelu)),
				  x-> leakyrelu.(x, 0.2f0),
				  x-> reshape(x, :, size(x, 4)),
				  Dropout(0.4),
				  SpectralNorm(Dense(6272, 1))) |> gpu
   MNISTDConvSpectral(d_labels, d_common)
end

function (m::MNISTDConvSpectral)(x, y)
	t = cat(m.d_labels(y), x, dims=3)
	return m.d_common(t)
end


## MNIST discriminator with dense layers
struct MNISTDMLP
	net
end

Flux.trainable(d::MNISTDMLP) = (d.net,)

function MNISTDMLP(s::Settings)
	MNISTDMLP(Chain(Dense(28*28 + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 1)) |> gpu)
end

function (m::MNISTDMLP)(x, y)
	m.net(vcat(reshape(x, :, size(y,2)), y))
end
	

## MNIST generator with conv layers
struct MNISTGConv
    g_labels          # Submodel to take labels as input and convert it to the shape of (7, 7, 1, batch_size) 
    g_latent          # Submodel to take latent_dims as input and convert it to shape of (7, 7, 128, batch_size)
    g_common    
end

Flux.trainable(d::MNISTGConv) = (d.g_labels, d.g_latent, d.g_common)

function MNISTGConv(s::Settings)
	g_labels = Chain(Dense(s.nclasses, 49), x-> reshape(x, 7 , 7 , 1 , size(x, 2))) |> gpu
    g_latent = Chain(Dense(s.latent_dim, 6272), x-> leakyrelu.(x, 0.2f0), x-> reshape(x, 7, 7, 128, size(x, 2))) |> gpu
    g_common = Chain(ConvTranspose((4, 4), 129=>128; stride=2, pad=1),
            BatchNorm(128, leakyrelu),
            Dropout(0.25),
            ConvTranspose((4, 4), 128=>64; stride=2, pad=1),
            BatchNorm(64, leakyrelu),
            Conv((7, 7), 64=>1, tanh; stride=1, pad=3)) |> gpu
	MNISTGConv(g_labels, g_latent, g_common)
end

function (m::MNISTGConv)(x, y)
    t = cat(m.g_labels(y), m.g_latent(x), dims=3)
    return m.g_common(t)
end

## MNIST generator with mlp
struct MNISTGMLP
	net
end

Flux.trainable(d::MNISTGMLP) = (d.net,)

function MNISTGMLP(s::Settings)
 	MNISTGMLP(Chain(Dense(s.latent_dim + s.nclasses, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 256, relu), Dense(256, 28*28, tanh), x -> reshape(x, 28,28,1,:)) |> gpu)
end
 
function (m::MNISTGMLP)(x, y)
	m.net(vcat(reshape(x, :, size(y,2)), y))
end

## Functions for generating outputs

# Function that returns random input for the generator
function MNIST_input(s::Settings)
	x = randn(s.latent_dim, s.batch_size) |> gpu
    y = Float32.(Flux.onehotbatch(rand(0:s.nclasses-1, s.batch_size), 0:s.nclasses-1)) |> gpu
    x, y
end

function gen_MNIST_images(s::Settings)
	# Load MNIST dataset
	images, labels = MLDatasets.MNIST.traindata(Float32)
	images = reshape(2f0 .* images .- 1f0, 28, 28, 1, :) |> gpu # Normalize to [-1, 1]
	y = Float32.(Flux.onehotbatch(labels, 0:s.nclasses-1)) |> gpu
	data = DataLoader((images, y), batchsize=s.batch_size, shuffle = true, partial = false)

	fixed_noise = [randn(s.latent_dim, 1) |> gpu for _=1:s.output_x * s.output_y]
	fixed_labels = [Float32.(Flux.onehotbatch(rand(0:s.nclasses-1, 1), 0:s.nclasses-1)) |> gpu 
							 for _ =1:s.output_x * s.output_y]
								 
	data, fixed_noise, fixed_labels
end

train(Settings(G = MNISTGConv, D = MNISTDConv, rand_input = MNIST_input, loss = DCGANLoss(), img_fun = gen_MNIST_images, nclasses = 10, output_dir = "MNIST_DCGAN"))
train(Settings(G = MNISTGConv, D = MNISTDConvSpectral, rand_input = MNIST_input, loss = DCGANLoss(), img_fun = gen_MNIST_images, nclasses = 10, output_dir = "MNIST_DCGAN_Spectral"))
train(Settings(G = MNISTGConv, D = MNISTDConv, rand_input = MNIST_input, loss = LSLoss(), img_fun = gen_MNIST_images, nclasses = 10, output_dir = "MNIST_LSGAN"))
train(Settings(G = MNISTGMLP, D = MNISTDMLP, rand_input = MNIST_input, loss = WLossGP(), img_fun = gen_MNIST_images, nclasses = 10, output_dir = "MNIST_WGAN_GP"))
