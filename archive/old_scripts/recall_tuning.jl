using BSON
using HDF5
using Plots

include("taxi_models_and_data.jl")
include("gan_evaluation.jl");

fn = "data/SK_DownsampledGANFocusAreaData.h5"
images = h5read(fn, "y_train")
real_images = reshape(images, 16*8, :)
y = h5read(fn, "X_train")[1:2, :];

# dirs = BSON.load("dirs.bson")[:dirs]
# generators = Dict(dir => todevice(BSON.load("$(dir)/conv_generator_ld2.bson")[:G], gpu) for dir in dirs)


dirs = first(walkdir("mlp_generator_results/"))[2]
generators = Dict(dir => todevice(BSON.load("mlp_generator_results/$(dir)/mlp_generator.bson")[:G], gpu) for dir in dirs)

# gconv2 = BSON.load("conv_generator_ld2.bson")[:G]
# gconv3_1 = BSON.load("dcgan_bce/conv_generator_ld3.bson")[:G]
# gconv3_2 = BSON.load("retrain/conv_generator_ld3.bson")[:G]
# gconv6 = BSON.load("conv_generator_ld6.bson")[:G]
# gconv12 = BSON.load("conv_generator_ld12.bson")[:G]
# gconv24 = BSON.load("conv_generator_ld24.bson")[:G]
# gconv50 = BSON.load("conv_generator_ld50.bson")[:G]
# gconv100 = BSON.load("conv_generator_ld100.bson")[:G]
# 
# mlp256x4_3 = BSON.load("shrinking_generator_results/three_ld/MLP256x4_mae_ld3.bson")[:G]
# mlp256x5_3 = BSON.load("shrinking_generator_results/three_ld/MLP256x5_mae_ld3.bson")[:G]
# mlp256x4_adv_3 = BSON.load("shrinking_generator_results/three_ld/MLP256x4-D_mae_ld3.bson")[:G]
# mlp256x5_adv_3 = BSON.load("shrinking_generator_results/three_ld/MLP256x5-D_mae_ld3.bson")[:G]


function vary_sample_size_conv(generator, sizes, real_images; k = 10, ld)
    # Get the fake images to cover all sizes
    N = maximum(sizes)
    lds, s = gpu.(get_inputs(N, ld=ld))
    fake_images = zeros(Float32, 16, 8, N)
    steps=0
    chunk = 10000
    while steps <= N-chunk
        r = steps+1:steps+chunk
        fake_images[:,:,r] .= dropdims(cpu(generator(lds[:,r], s[:,r])), dims=3)
        steps += chunk
        println("steps: ", steps)
    end
    fake_images = (reshape(fake_images, 128, :) .+ 1f0) ./ 2f0

    # Vary the size
    recalls = zeros(length(sizes))
    for i = 1:length(sizes)
        #println(i)
        recalls[i] = recall(fake_images[:, 1:sizes[i]], real_images, k = k)
    end

    return recalls
end

sizes = [100, 10000, 30000, 50000];
recall_dict = Dict(d => vary_sample_size_conv(generators[d], sizes, real_images, ld=2) for d in dirs)

BSON.@save "recall_results_all.bson" recall_dict

using Plots
p = plot(legend = :bottomleft)
for d in dirs
    plot!(recall_dict[d], label=d)
end
p
savefig("all_results.pdf")

recall_two_conv50_50 = vary_sample_size_double_conv(gconv3_1, gconv3_2, sizes, real_images; k=30, ld=3)

recall_dict["BCE_BS256_LR0.0007"]

recalls_conv2 = vary_sample_size_conv(gconv2, sizes, real_images; k=30, ld=2)
recalls_conv3 = vary_sample_size_conv(gconv3, sizes, real_images; k=30, ld=3)
recalls_conv6 = vary_sample_size_conv(gconv6, sizes, real_images; k=30, ld=6)
recalls_conv12 = vary_sample_size_conv(gconv12, sizes, real_images; k=30, ld=12)
recalls_conv24 = vary_sample_size_conv(gconv24, sizes, real_images; k=30, ld=24)
recalls_conv50 = vary_sample_size_conv(gconv50, sizes, real_images; k=30, ld=50)
recalls_conv100 = vary_sample_size_conv(gconv100, sizes, real_images; k=30, ld=100)

recalls_mlp256x4_3 = vary_sample_size_conv(mlp256x4_3, sizes, real_images; k=30, ld=3)
recalls_mlp256x5_3 = vary_sample_size_conv(mlp256x5_3, sizes, real_images; k=30, ld=3)
recalls_mlp256x4_adv_3 = vary_sample_size_conv(mlp256x4_adv_3, sizes, real_images; k=30, ld=3)
recalls_mlp256x5_adv_3 = vary_sample_size_conv(mlp256x5_adv_3, sizes, real_images; k=30, ld=3)


# recalls = Dict(:sizes => sizes, :recalls_conv2=>recalls_conv2, :recalls_conv3=>recalls_conv3, :recalls_conv6=>recalls_conv6, :recalls_conv12=>recalls_conv12, :recalls_conv24=>recalls_conv24, :recalls_conv50=>recalls_conv50, :recalls_conv100=>recalls_conv100)

recalls = BSON.load("dcgan_bce/recall_data.bson")[:recalls]

recalls[:recalls_mlp256x4_3] = recalls_mlp246x4_3
recalls[:recalls_mlp256x5_3] = recalls_mlp246x5_3
recalls[:recalls_mlp256x4_adv_3] = recalls_mlp246x4_adv_3
recalls[:recalls_mlp256x5_adv_3] = recalls_mlp246x5_adv_3

BSON.@save "recall_data.bson" recalls


# Plots.plot(sizes, recalls[:recalls_conv2], marker=true, label="2 ld")
Plots.plot(sizes, recalls[:recalls_conv3], marker=true, label="3 ld")
Plots.plot!(sizes, recall_two_conv, marker=true, label="3 ld - 2 generators (2/3 -1/3)")
Plots.plot!(sizes, recall_two_conv50_50, marker=true, label="3 ld - 2 generators (1/2 - 1/2)")
savefig("two_generator_recall.pdf")
# Plots.plot!(sizes, recalls_conv6, marker=true, label="6 ld")
# Plots.plot!(sizes, recalls_conv12, marker=true, label="12 ld")
# Plots.plot!(sizes, recalls_conv24, marker=true, label="24 ld")
# Plots.plot!(sizes, recalls_conv50, marker=true, label="50 ld")
# Plots.plot!(sizes, recalls_conv100, marker=true, label="100 ld")
Plots.plot!(sizes, recalls[:recalls_mlp256x4_3], marker=true, label="256x4")
Plots.plot!(sizes, recalls[:recalls_mlp256x5_3], marker=true, label="256x5")
Plots.plot!(sizes, recalls[:recalls_mlp256x4_adv_3], marker=true, label="256x4 - adversarial")
Plots.plot!(sizes, recalls[:recalls_mlp256x5_adv_3], marker=true, label="256x5 - adversarial")


Plots.savefig("recall_vs_cpmpression.pdf")



# function vary_sample_size_double_conv(generator1, generator2, sizes, real_images; k = 10, ld)
#     # Get the fake images to cover all sizes
#     N = maximum(sizes)
#     lds, s = get_inputs(maximum(sizes), ld=ld)
#     fake_images = zeros(Float32, 16, 8, N)
#     for i = 1:N
#         i%1000 == 0 && println(i)
#         if rand() < 0.5
#             fake_images[:, :, i] = generator1(lds[:, i], s[:, i])[:, :, 1, 1]
#         else
#             fake_images[:, :, i] = generator2(lds[:, i], s[:, i])[:, :, 1, 1]
#         end
#     end
#     fake_images = (reshape(fake_images, 128, :) .+ 1f0) ./ 2f0
# 
#     # Vary the size
#     recalls = zeros(length(sizes))
#     for i = 1:length(sizes)
#         #println(i)
#         recalls[i] = recall(fake_images[:, 1:sizes[i]], real_images, k = k)
#     end
# 
#     return recalls
# end
