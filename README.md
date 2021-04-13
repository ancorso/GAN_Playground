# Code for generating GANs

Here is a short description of the important files
* `cGAN_common.jl` contains all the code for training our conitional GANs
* `taxi_models_and_data.jl` contains all the code for creating the generator and discriminator models as well as loading in the data
* `train_gans.jl` has the code for actually training the gans (With a variety of hyperparameters)
* `data\` contains the datasets we use for the taxi problem
* `train_smaller_generator.jl` contains the code for the supervised training of the smaller generator.
