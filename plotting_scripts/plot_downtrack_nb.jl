### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 4a974030-7c3f-11eb-3b37-abee20c31afd
using HDF5, Plots, Statistics, Base.Iterators, PlutoUI, Images, BSON, Flux

# ╔═╡ 5779b29c-7c3f-11eb-1017-0b9b3372faae
fn = "data/SK_DownsampledGANFocusAreaData.h5";

# ╔═╡ 6d1ca366-7c3f-11eb-18ea-97a467f1d860
images = h5read(fn, "y_train");

# ╔═╡ 6e06efde-7c3f-11eb-0c63-d59c70c39e12
y = h5read(fn, "X_train");

# ╔═╡ 6e704ad8-7c3f-11eb-2dee-ad2d7b120383
down_start = y[3,1];

# ╔═╡ 6f74c7c4-7c3f-11eb-2af1-555561d0b376
dash_distance = 30.45;

# ╔═╡ 8968a3e2-7c3f-11eb-3de1-e14b37a647ef
begin
	y[1,:] ./= std(y[1,:])
	y[2,:] ./= std(y[2,:])
	y[3,:] .= rem.((y[3,:] .- down_start), dash_distance)
	y[3,:] .= (y[3,:] .- mean(y[3,:]))./std(y[3,:])
end;

# ╔═╡ b870935e-7c3f-11eb-0f0d-e34dc392d384
indices = findall((y[1, :] .> -0.1) .& (y[1,:] .< 0.1) .& (y[2,:] .> -0.1) .& (y[2,:] .< 0.1));

# ╔═╡ a93b98b0-7c40-11eb-1595-71757405d980
md"**Check that downtrack variable looks smooth**"

# ╔═╡ c2a06cbe-7c3f-11eb-15ba-8505170fffda
begin
	dt = y[3,indices]
	real_images = [images[:,:,i] for i in indices]

	order = sortperm(dt)
	real_images = real_images[order]
end;

# ╔═╡ d563fe4e-7c3f-11eb-2ff1-f3ff42f7527a
@bind i Slider(1:length(indices))

# ╔═╡ 613f6c78-7c40-11eb-102f-8bf4a8873480
begin
	p1 = plot(Gray.(real_images[i])', size=(400,200))
	p2 = plot(dt[order], title="Downtrack Variable", label="", size=(200,200))
	scatter!([i], dt[order[i:i]],label="")
	plot(p1, p2, size=(600,200))
end

# ╔═╡ bb88694e-7c40-11eb-2f9c-c5186f39d977
md"**Check Consistency accross downtrack variable**"

# ╔═╡ 9522e240-7c43-11eb-098a-1fd07b963ce2
desired_downtrack = 1.;

# ╔═╡ 9000c690-7c45-11eb-02a6-dde41fb17673
Δ = 0.02;

# ╔═╡ a2e4ebd8-7c43-11eb-1f84-f9ab10fc4ba6
match_indices = findall((y[3,:] .> desired_downtrack - Δ) .& (y[3,:] .< desired_downtrack + Δ));

# ╔═╡ bac43230-7c44-11eb-1769-d1f2d809ce65
length(match_indices)

# ╔═╡ 7943432a-7c44-11eb-22b3-51feb83d8bee
@bind k Slider(1:length(match_indices))

# ╔═╡ 87563472-7c44-11eb-399b-bfe7c7062b5a
begin
	mi = match_indices[k]
	p3 = plot(Gray.(images[:,:,mi])', size=(400,200))
	p4 = scatter(y[1,match_indices], y[2,match_indices], xlims=(-2,2), ylims=(-2,2), marker=true, markersize=3, size=(200,200), label="", xlabel="cte", ylabel="he", alpha=0.2)
	scatter!([y[1,mi]], [y[2,mi]], label="")
	plot(p3,p4, size=(600,200))
end
	

# ╔═╡ Cell order:
# ╠═4a974030-7c3f-11eb-3b37-abee20c31afd
# ╠═5779b29c-7c3f-11eb-1017-0b9b3372faae
# ╠═6d1ca366-7c3f-11eb-18ea-97a467f1d860
# ╠═6e06efde-7c3f-11eb-0c63-d59c70c39e12
# ╠═6e704ad8-7c3f-11eb-2dee-ad2d7b120383
# ╠═6f74c7c4-7c3f-11eb-2af1-555561d0b376
# ╠═8968a3e2-7c3f-11eb-3de1-e14b37a647ef
# ╠═b870935e-7c3f-11eb-0f0d-e34dc392d384
# ╠═a93b98b0-7c40-11eb-1595-71757405d980
# ╠═c2a06cbe-7c3f-11eb-15ba-8505170fffda
# ╠═d563fe4e-7c3f-11eb-2ff1-f3ff42f7527a
# ╠═613f6c78-7c40-11eb-102f-8bf4a8873480
# ╟─bb88694e-7c40-11eb-2f9c-c5186f39d977
# ╠═9522e240-7c43-11eb-098a-1fd07b963ce2
# ╠═9000c690-7c45-11eb-02a6-dde41fb17673
# ╠═a2e4ebd8-7c43-11eb-1f84-f9ab10fc4ba6
# ╠═bac43230-7c44-11eb-1769-d1f2d809ce65
# ╠═7943432a-7c44-11eb-22b3-51feb83d8bee
# ╠═87563472-7c44-11eb-399b-bfe7c7062b5a
