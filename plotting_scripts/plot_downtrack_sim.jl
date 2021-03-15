using HDF5, Plots, Statistics
import Base.Iterators:partition

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


indices = findall((y[1, :] .> -0.1) .& (y[1,:] .< 0.1) .& (y[2,:] .> -0.1) .& (y[2,:] .< 0.1))
dt = y[3,indices]
real_images = [images[:,:,i] for i in indices]

order = sortperm(dt)
real_images = real_images[order]

plot(dt[order])

# visualize the images
nrows = 39
image_array = permutedims(reduce(vcat, reduce.(hcat, partition(real_images, nrows))), (2,1))
image_array = Gray.(image_array)

