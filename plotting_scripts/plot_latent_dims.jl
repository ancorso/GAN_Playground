using BSON
include("taxi_models_and_data.jl")

G = BSON.load("downtrack_generator_ld2.bson")[:G]

nd = 2
s = [[0, 0, 2*(i-1) / 36 - 1] for i=1:36]
ld = [zeros(nd) for i=1:36]
output = G.(ld, s)

Gray.((permutedims(dropdims(reduce(hcat, output), dims = (3,4)), (2,1)) .+1f0) ./ 2f0)

image_array = permutedims(dropdims(reduce(vcat, reduce.(hcat, partition(output, 6))); dims=(3, 4)), (2, 1))
image_array = Gray.(image_array .+ 1f0) ./ 2f0

save("mode_collapse.png", image_array)

