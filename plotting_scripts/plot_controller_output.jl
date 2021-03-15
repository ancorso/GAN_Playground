using Flux, BSON, HDF5, Plots
# include("taxi_models_and_data.jl")

m = BSON.load("concatenated_controllers/full_big_normal_v2.bson")[:model]

input = vcat(randn(2,1000), rand(Uniform(-1.72,1.72),2,1000))
out = m(input)

scatter!(input[3,:]*6.366468343804353, out[1,:], label="generated images (Big MLP)", color = :black)

savefig("generated_vs_real_image_error.png")


# labels = randn(2,1000)
# fake_images = G_dcgan2(randn(2,1000) |>gpu,  labels|>gpu) |>cpu
# data = Dict(:images => fake_images, :labels => labels)
# BSON.@save "dcgan_generated_images.bson" data
data = BSON.load("dcgan_generated_images.bson")[:data]


taxinet = BSON.load("concatenated_controllers/taxinet.bson")[:tn]

fn = "data/SK_DownsampledGANFocusAreaData.h5"
images = h5read(fn, "y_train") # yes I know the labels seem backwards
println("starting size: ", size(images))
println("permuted dims: ", size(images))
y = h5read(fn, "X_train")[1:2, :]
    println("std1: ", std(y[1,:]), " std2: ", std(y[2,:]))
y[1,:] ./= std(y[1,:])
y[2,:] ./= std(y[2,:])

real_pred = taxinet(reshape(images[:,:, 1:1000], :, 1000))

scatter(y[1,1:1000]*6.366468343804353, real_pred[1,:], xlabel = "Real", ylabel="Prediction", title = "CTE", label = "real images", legend = :bottomright)


fake_images = data[:images]
labels = data[:labels]
fake_pred = taxinet(reshape((fake_images .+ 1) ./ 2, :, 1000))
scatter!(fake_pred[1,:], labels[1,:]*6.366468343804353, color = :black, label = "dc_gan", )


Gbigmlp = BSON.load("generators/bigmlp_generator_normalnoise_permuted.bson")[:Gbig]
labels2 = randn(2,1000)
fake_images2 = Gbigmlp(vcat(rand(Uniform(-2, 2), 2,1000),  labels2))

fake_pred2 = taxinet(reshape((fake_images2 .+ 1) ./ 2, :, 1000))
scatter!(fake_pred2[1,:], labels2[1,:]*6.366468343804353, label = "bigmlp")



Gsmallmlp = BSON.load("generators/smallmlp_generator_normalnoise.bson")[:Gsmall]
labels3 = rand(Uniform(-2,2),2,1000)
fake_images3 = Gbigmlp(vcat(randn(2,1000),  labels3))

fake_pred3 = taxinet(reshape((fake_images3 .+ 1) ./ 2, :, 1000))
scatter!(fake_pred3[1,:], labels3[1,:]*6.366468343804353, color = :black, label = "smallmlp")

