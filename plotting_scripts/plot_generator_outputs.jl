include("../taxi_models_and_data.jl")
using BSON

Gmlp = BSON.load("mlp_generator_results/MLP256x4_mae_adv_ld2_λ0.001_LR0.007/mlp_generator.bson")[:G]

Gray.(reshape((Gmlp([-0.99969482421875, 0.99969482421875], [-3.067831165611182e-5, 0.019391002053608518]) .+ 1) ./ 2, 16,8))'

#Requires Flux#641d86796cbdf6e957e15fa4e43ce1efd73c5790
Gconv = BSON.load("conv_generator_results/BCE_BS256_LR0.0007/conv_generator_ld2.bson")[:G]


Gray.(reshape((Gconv([-0.99969482421875, 0.99969482421875], [-3.067831165611182e-5, 0.019391002053608518]) .+ 1) ./ 2, 16,8))'

