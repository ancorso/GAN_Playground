using BSON
include("taxi_models_and_data.jl")

nd = 100
G = BSON.load("generators/dcgan_ld$(nd)_generator_permuted.bson")[:G]

s = [zeros(2) for i=1:36]
ld = [randn(nd) for i=1:36]
output = G.(ld, s)
output = permutedims.(output, [[2,1,3,4]])

image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(output, 6))); dims=(3, 4)), (2, 1))
image_array = Gray.(image_array .+ 1f0) ./ 2f0

