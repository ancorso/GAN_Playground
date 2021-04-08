using Flux, Statistics, MLDatasets, Random
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using CUDA

function mlp_model(imgsize = (28,28,1), nclasses = 10)
    return Chain(Dense(prod(imgsize), 32, relu), Dense(32,32, relu), Dense(32, nclasses)) |> gpu
end

function conv_model(imgsize = (28,28,1), nclasses = 10)
    cnn_output_size = Int.(floor.([imgsize[1]/8,imgsize[2]/8,32]))	

    return Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), imgsize[3]=>16, pad=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    MaxPool((2,2)),

    # Reshape 3d tensor into a 2d one using `Flux.flatten`, at this point it should be (3, 3, 32, N)
    flatten,
    Dense(prod(cnn_output_size), 10)) |> gpu
end

augment(x) = x .+ gpu(0.1f0*randn(Float32, size(x)))

compare(y, y′) = maximum(y′, dims = 1) .== maximum(y .* y′, dims = 1)
accuracy(model, x, y) = mean(compare(y, cpu(model)(x)))

function gen_data(;permute = true, seed = 0, flatten = true, train_on_gpu = true, test_on_gpu = false, imgsize = (28,28,1))
    train_x, train_y = MNIST.traindata(Float32)
    test_x, test_y  = MNIST.testdata(Float32)
    
    N = prod(imgsize)
    permutation = (permute) ? randperm(MersenneTwister(seed), N) : 1:N

    train_x = reshape(train_x, N, :)[permutation, :]
    train_y = Float32.(Flux.onehotbatch(train_y, 0:9))
    test_x = reshape(test_x, N, :)[permutation, :]
    test_y = Float32.(Flux.onehotbatch(test_y, 0:9)) 
    
    if !flatten
        train_x = reshape(train_x, imgsize..., :)
        test_x = reshape(test_x, imgsize..., :)
    end 
    if train_on_gpu
        train_x = train_x |> gpu
        train_y = train_y |> gpu
    end
    if test_on_gpu
        test_x = test_x |> gpu
        test_y = test_y |> gpu
    end
    train_x, train_y, test_x, test_y
end

function train(model, train_x, train_y, test_x, test_y; nclasses = 10, lr::Float64 = 1e-3, epochs::Int = 20, batch_size = 128, savepath::String = "./", opt = ADAM(lr))	
    data = DataLoader((train_x, train_y), batchsize = batch_size, shuffle = true)
	
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:epochs
        # Train for a single epoch
        Flux.train!((x,y) -> logitcrossentropy(model(augment(x)), y), params(model), data, opt)
	
        # Calculate accuracy:
        acc = accuracy(model, test_x, test_y)
        println("[$epoch_idx]: Test accuracy: $acc")

	
        # If we haven't seen improvement in 5 epochs, drop our learning rate:
        if epoch_idx - last_improvement >= 10 && opt.eta > 1e-6
            opt.eta /= 10.0
            @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
        
            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end
    end
end



tr_x, tr_y, te_x, te_y = gen_data(permute = false)
mlp = mlp_model()
train(mlp, tr_x, tr_y, te_x, te_y)

fixed_noise = randn(hp.latent_dim, 100) |> gpu
fixed_labels = Float32.(Flux.onehotbatch(zeros(100), 0:hp.nclasses-1)) |> gpu 

input = (G(fixed_noise, fixed_labels) .+ 1.0f0) / 2.0f0
Gray.(cpu(input[:,:,1,1]))

compare(mlp(flatten(input)), fixed_labels)

y = mlp(flatten(input))
mean(maximum(fixed_labels .* y, dims = 1) .== maximum(y, dims = 1))

Flux.onehot(0, 0:9)



