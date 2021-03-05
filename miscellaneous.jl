using Plots,Statistics,Distributed,Distributions,ColorSchemes,Hungarian,JLD,SpecialFunctions,LaTeXStrings,Dates
pyplot(box=true,label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3) ; plot()
include("methods.jl");

## Different representations of the system
cols = [:white,:blue,:green,:orange,:red]
Var,T,L,init,tmax,lattice = load("data\\XY_realisation.jld","Var","T","L","init","tmax","lattice")
Var,T,L,init,tmax,lattice = load("data\\DeXY_realisation.jld","Var","T","L","init","tmax","lattice")

Var = 0.1 ; T = 0. ; L = 200 ; init = "HighTemp" ; tmax = 6000 ; t = 0.0
lattice,dt = create_Lattice(L,T,Var,init)

z = @elapsed while t < tmax
    global lattice = update(lattice,T,L,dt)
    global t = t + dt
end

thetas=lattice[:,:,1]
a=heatmap(mod.(thetas,2pi)',c=cgrad([:white,:blue,:green,:orange,:red,:white]),title=L"\theta",size=(620,500))
b=heatmap(diffops(mod.(thetas,2pi))[1]',c=cgrad([:blue,:green,:orange,:red]),title=L"\nabla \theta",size=(620,500),clims=(0,1))
c=heatmap(psi(thetas)',c=cgrad([:darkblue,:dodgerblue3,:forestgreen,:darkgreen,:orange,:red2]),title=L"\psi",size=(620,500),clims=(-1,1))
d=heatmap(transpose(diffops(psi(thetas))[1].^1),c=cgrad([:white,:blue,:green,:orange,:red]),title=L"\nabla\psi",size=(620,500),clims=(0,1))
e=heatmap(energy(thetas)[1]' .+4,title=L" E",size=(620,500),clims=(0,1.5))
f=heatmap(diffops(energy(thetas)[1] .+4)[1]',c=cgrad([:white,:blue,:green,:orange,:red]),title=L"\nabla E",size=(620,500),clims=(0,0.5))
plot(a,b,c,d,e,f,layout=(3,2),size=(1240,1500))
savefig("figures\\representations_XY.pdf")
savefig("figures\\representations_XY.png")
savefig("figures\\representations_NKM.pdf")
savefig("figures\\representations_NKM.png")


## Verify that there is no anisotropy in the correlation function C(r)
    # For the XY Model  : OK, done on the 14.10.2020
    # For the SRK Model : OK, done on the 15.10.2020
    # For the QLK Model : OK, done on the 15.10.2020
begin
    using Plots,Statistics,Distributed,Distributions,ColorSchemes
    pyplot(box=true,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3) ; plot()
    include("methods.jl");

    Var = 0.25 ; T = 0.2 ; L = 100 ; init = "LowTemp" ; tmax = 200 ; t = 0.0
    lattice_old,dt = create_Lattice(L,T,Var,init)
    lattice_new    = copy(lattice_old);

    z = @elapsed while t < tmax
        global lattice_new = update(lattice_old,lattice_new,T,L,dt)
        global lattice_old = lattice_new
        global t = t + dt
    end

    C_vertical , C_horizontal , C_diag , C_anti_diag = C_all_directions(lattice_new)
    C_all = get_C(lattice_new[:,:,1])

    plot(xlabel="r",ylabel="C(r)",title="L = $L , σ² = $Var , T=$T")
    plot!(1:50,C_vertical,axis=:lin,label="Vertical",ribbon=0)
    plot!(1:50,C_horizontal,axis=:lin,label="Horizontal",ribbon=0)
    plot!(Array(1:50) * sqrt(2),C_diag,axis=:lin,label="Diagonal",ribbon=0)
    plot!(Array(1:50) * sqrt(2),C_anti_diag,axis=:lin,label="Anti Diagonal",ribbon=0)
    plot!(sort(vcat(1:50,Array(1:50) * sqrt(2))),smooth(C_all),axis=:lin,label="All Combined",ribbon=0)
    savefig("no_anisotropy_QLK.pdf")
end

## After reading the PRE by Lee et al, 2010
    #= I want to verify the presence of a macro-cluster of
    effective frequencies if σ < σ_E ≈ 0.45 =#
Var = 0.0 ; T = 0.0 ; L = 200 ; init = "LowTemp" ; tmax = 200 ; t = 0.0 ; t1 = 100
lattice_old,dt = create_Lattice(L,T,Var,init)
lattice_new    = copy(lattice_old);

z = @elapsed while t < t1
    global lattice_new = update(lattice_old,lattice_new,T,L,dt)
    global lattice_old = lattice_new
    global t = t + dt
end
""
    saved_lattice = copy(lattice_new)
    t
    while t < tmax
        global lattice_new = update(lattice_old,lattice_new,T,L,dt)
        global lattice_old = lattice_new
        global t = t + dt
    end
grille = [(i,j) for i=1:L, j=1:L]
xs   = [grille[n][2] for n in 1:length(grille)]  ; ys   = [grille[n][1] for n in length(grille):-1:1]
effective_frequency = (lattice_new[:,:,1] .- saved_lattice[:,:,1])/(tmax-t1)
p = heatmap(effective_frequency,c=cgrad([:blue,:green,:orange,:red]),title="L = $L , σ² = $Var , T=$T")
quiver!(xs,ys,quiver=(vec(cos.(lattice_new[:,:,1])),vec(sin.(lattice_new[:,:,1]))),label=false,yticks=(0:10:L,L:-10:0),xticks=(0:10:L),color=:white)
vortices = spot_vortices(lattice_new[:,:,1])
for vortex in vortices
    if vortex[3] == +1 mcol = :black ; mstrcol=:black ; mswidth = 2 # full circle
    else mcol = :white ; mstrcol=:black ; mswidth = 2 # empty circle
    end
    scatter!(correct_location(vortex[1:2],L).+(0.5,0.5),m=:circle,color=mcol,msc=mstrcol,msw=mswidth,ms=:10,label=nothing)
end
display(p)
savefig("figures\\angles_freq_L$(L)_Var$(Var)_T$(T)_init$(init).pdf")
""
histogram(vec(effective_frequency),yaxis=:log,normalize=:pdf)

## Running Time Experiments
# Update Method
@everywhere function update(lattice_old::Array{Float64,3},lattice_new::Array{Float64,3},T::Number,L::Int,dt::Float64)::Array{Float64,3}
    for j in 1:L
        for i in 1:L
            angle_neighbours = get_angle_nearest_neighbours(lattice_old[:,:,1],i,j,L) # O(1)
            lattice_new[i,j,1] = lattice_old[i,j,1] + lattice_old[i,j,2]*dt + dt*sum(Float64[sin(angle - lattice_old[i,j,1]) for angle in angle_neighbours]) + sqrt(2*T*dt)*randn() # O(1)
        end
    end
    return lattice_new
end
@everywhere function update1(lattice_old::Array{Float64,3},lattice_new::Array{Float64,3},T::Number,L::Int,dt::Float64)::Array{Float64,3}
    for j in 1:L
        for i in 1:L
            angle_neighbours = get_angle_nearest_neighbours(lattice_old[:,:,1],i,j,L) # O(1)
            a,b = lattice_old[i,j,:]
            lattice_new[i,j,1] = a + b*dt + dt*sum(Float64[sin(angle - a) for angle in angle_neighbours]) + sqrt(2*T*dt)*randn() # O(1)
        end
    end
    return lattice_new
end
@everywhere function update2(lattice_old::Array{Float64,3},lattice_new::Array{Float64,3},T::Number,L::Int,dt::Float64)::Array{Float64,3}
    for i in 1:L
        for j in 1:L
            angle_neighbours = get_angle_nearest_neighbours(lattice_old[:,:,1],i,j,L) # O(1)
            lattice_new[i,j,1] = lattice_old[i,j,1] + lattice_old[i,j,2]*dt + dt*sum(Float64[sin(angle - lattice_old[i,j,1]) for angle in angle_neighbours]) + sqrt(2*T*dt)*randn() # O(1)
        end
    end
    return lattice_new
end
@everywhere function update3(lattice_old::Array{Float64,3},lattice_new::Array{Float64,3},T::Number,L::Int,dt::Float64)::Array{Float64,3}
    for j in 1:L
        for i in 1:L
            #= NB : the inversion j,i is wanted. Because Julia accesses the matrices by columns and not by rows,
            for i,j leads to O(L^4) while for j,i leads to O(L^2), as expected.  =#
            angle_neighbours = get_angle_nearest_neighbours(lattice_old[:,:,1],i,j,L) # O(1)
            a,b = lattice_old[i,j,:]
            sin_angle_neighbours = sin.(angle_neighbours .- a)
            lattice_new[i,j,1] =  a + b*dt + dt*sum(sin_angle_neighbours) + sqrt(2*T*dt)*randn() # O(1)
        end
    end
    return lattice_new
end
@everywhere function update4(lattice_old::Array{Float64,3},lattice_new::Array{Float64,3},T::Number,L::Int,dt::Float64)::Array{Float64,3}
    noise = randn(L,L)
    thetas = lattice_old[:,:,1]
    omegas = lattice_old[:,:,2]
    thetas_new = Array{Float64,2}(undef,L,L)
    for j in 1:L
        for i in 1:L
            #= NB : the inversion j,i is wanted. Because Julia accesses the matrices by columns and not by rows,
            for i,j leads to O(L^4) while for j,i leads to O(L^2), as expected.  =#
            angle_neighbours = get_angle_nearest_neighbours(thetas,i,j,L) # O(1)
            θ,ω = thetas[i,j] , omegas[i,j]
            sin_angle_neighbours = sin.(angle_neighbours .- θ)
            thetas_new[i,j] =  θ + ω*dt + dt*sum(sin_angle_neighbours) + sqrt(2*T*dt)*noise[i,j] # O(1)
        end
    end
    return cat(thetas_new, omegas, dims=3)
end

Ls = [10,20,30,50,75,100,150,200,300,400,500]
z = zeros(5,length(Ls))
for i in eachindex(Ls)
    L = Ls[i]
    println("L=$L")
    T = Var = 0.0
    lattice_old,dt = create_Lattice(L,T,Var,init)
    lattice_new    = copy(lattice_old);
    # z[1,i] = @elapsed for i in 1:50 lattice_new = update(lattice_old,lattice_new,T,L,dt) ; lattice_old = lattice_new end
    # z[2,i] = @elapsed for i in 1:50 lattice_new = update1(lattice_old,lattice_new,T,L,dt) ; lattice_old = lattice_new end
    # z[3,i] = @elapsed for i in 1:50 lattice_new = update2(lattice_old,lattice_new,T,L,dt) ; lattice_old = lattice_new end
    # z[4,i] = @elapsed for i in 1:50 lattice_new = update3(lattice_old,lattice_new,T,L,dt) ; lattice_old = lattice_new end
    z[5,i] = @elapsed for i in 1:50 lattice_new = update(lattice_old,lattice_new,T,L,dt) ; lattice_old = lattice_new end
end
sum(z)
p = plot(xlabel="L",ylabel="CPU time for 50 updates")
for i in 5:5
    plot!(Ls,z[i,:],label="Method #$(i-1)",axis=:log)
end
plot!(Ls,1E-5*Ls .^ 2)
display(p)



## Dans l'optique de train un NN, il faut générer des configurations à l'état stationnaire
    # Commençons par un simple XY model à T=0
@everywhere function create_Trainset(L,M)
    lattices = Array{Float16,3}(undef,L,L,M)
    for m in 1:M
        Var = 0.0 ; T = 0.0 ; init = "HighTemp" ; tmax = 150 ; t = 0.0

        lattice_old,dt = create_Lattice(L,T,Var,init)
        lattice_new    = copy(lattice_old);
        z = @elapsed while t < tmax
            lattice_new = update(lattice_old,lattice_new,T,L,dt)
            lattice_old = lattice_new
            t = t + dt
        end
        lattices[:,:,m] = Float16.(lattice_new[:,:,1])
    end
    return lattices
end

L = 32 ; Var = 0 ; T = 0 ; init = "HighTemp" ; tmax = 300 ;
nb_samples = 10000
lattices = Array{Float32,3}(undef,nb_samples+1,L,L)
nb_now = 0
@time while nb_now ≤ nb_samples
    lattice,dt = create_Lattice(L,T,Float64(Var),init)
    t = 0.0
    while t < tmax
        lattice = update(lattice,T,L,dt)
        t = t + dt
    end
    thetas = lattice[:,:,1]
    if length(spot_vortices(thetas)) > 0
        global nb_now += 1
        theta_modified = (mod.(thetas,2π) .- π) / π   # -1 ≤ data ≤ +1
        lattices[nb_now,:,:] = theta_modified
        # push!(lattices,theta_modified)
        print("$nb_now  ")
    end
end

# L,lattices,nb_samples,tmax,T,Var,init = JLD.load("ML_trainset_T0_Var0_L30_M10000.jld","L","lattices","nb_samples","tmax","T","Var","init")

Base.summarysize(lattices)/1E6 # size in Mo = 16*L²*M/1E6/8
h5open("data\\ML_trainset_T$(T)_Var$(Var)_L$(L)_M$nb_samples.h5", "w") do file
    HDF5.write(file,"L",L,"lattices",lattices,"nb_samples",nb_samples,"tmax",tmax,"T",T,"Var",Var,"init",init)
end

lattices_GAN = h5read("data/output_GAN.hdf5", "default")



gr()
grille = [(i,j) for i=1:L, j=1:L]
xs   = [grille[n][2] for n in 1:length(grille)]  ; ys   = [grille[n][1] for n in length(grille):-1:1]
ii = 3
quiver(xs,ys,quiver=(vec(cos.(pi .+ pi*lattices_GAN[:,:,ii])),vec(sin.(pi .+ pi*lattices_GAN[:,:,ii]))),label=false,yticks=(0:10:L,L:-10:0),xticks=(0:10:L),color=:grey)
quiver(xs,ys,quiver=(vec(cos.(pi .+ pi*lattices[ii,:,:])),vec(sin.(pi .+ pi*lattices[ii,:,:]))),label=false,yticks=(0:10:L,L:-10:0),xticks=(0:10:L),color=:grey)

## PCA on a lattice to see what gets out
# using PyCall
# pyplot()
# np = pyimport("numpy")
# plt = pyimport("matplotlib.pyplot")
# PCA = pyimport("sklearn.decomposition")
#
# img = np.mean(plt.imread("signature.jpg"),-1)
# img = img / 255
#
# imshow(img,cmap="gray")
# savefig(p,"image.png")
using MultivariateStats,JLD
img = testimage("lighthouse")
img = convert(Array{Float64},Gray.(img))
summary(img)

data = load("data\\trainset_L30_R1000.jld")
images = data["lattices"] ; L = data["L"]
    img = Float64.(images[:,:,2])

    grille = [(i,j) for i=1:L, j=1:L]
    xs   = [grille[n][2] for n in 1:length(grille)]  ; ys   = [grille[n][1] for n in length(grille):-1:1]
    quiver(xs,ys,quiver=(vec(cos.(img)),vec(sin.(img))),label=false,yticks=(0:10:L,L:-10:0),xticks=(0:10:L),color=:grey)

M = fit(PCA, img,pratio = 0.999)
Yte = transform(M, img)
Xr = reconstruct(M, Yte)
quiver(xs,ys,quiver=(vec(cos.(Xr)),vec(sin.(Xr))),label=false,yticks=(0:10:L,L:-10:0),xticks=(0:10:L),color=:grey)

lattice_new = zeros(L,L,2)
lattice_new[:,:,1] = Xr
lattice_old = lattice_new
T = 0.0
dt = 0.05
for i in 1:500
    global lattice_new = update(lattice_old,lattice_new,T,L,dt)
    global lattice_old = lattice_new
end
quiver(xs,ys,quiver=(vec(cos.(lattice_new[:,:,1])),vec(sin.(lattice_new[:,:,1]))),label=false,yticks=(0:10:L,L:-10:0),xticks=(0:10:L),color=:grey)

## Test number_free_vortex function & Visualisation
using Plots,Statistics,Distributed,Distributions,ColorSchemes,Hungarian
include("methods.jl");

Var = 1 ; T = 0 ; L = 30 ; init = "LowTemp" ; tmax = 100 ;
lattice_old,dt = create_Lattice(L,T,Var,init)
lattice_new    = copy(lattice_old);

t = 0.0
while t < tmax
    global lattice_new = update(lattice_old,T,L,dt)
    global lattice_old = lattice_new
    global t = t + dt
end
label_vortex,label_antivortex,number_free_vortex = classify_vortices(lattice_new[:,:,1])

    grille = [(i,j) for i=1:L, j=1:L]
    xs   = [grille[n][2] for n in 1:length(grille)]  ; ys   = [grille[n][1] for n in length(grille):-1:1]

    p = plot(yticks=(0:10:L,L:-10:0),xticks=(0:10:L),size=(500,500),title="T = $T , σ² = $Var")
    quiver!(xs,ys,quiver=(vec(cos.(lattice_new[:,:,1])),vec(sin.(lattice_new[:,:,1]))),label=false,color=:grey)
    list_vortex,list_antivortex  = spot_vortex_antivortex(lattice_new[:,:,1]) # Vector{Tuple{Int,Int}}
    for i in eachindex(list_vortex)
        if label_vortex[i] == 0 # if free
            mcol = :black ; mstrcol=:black ; mswidth = 2
        else # if bounded
            mcol = label_vortex[i] ; mstrcol=label_vortex[i] ; mswidth = 2
        end
        scatter!(correct_location(list_vortex[i],L).+(0.5,0.5),m=:circle,color=mcol,msc=mstrcol,msw=mswidth,ms=:10,label=nothing)
    end
    for i in eachindex(list_antivortex)
        if label_antivortex[i] == 0 # if free
            mcol = :white ; mstrcol=:black ; mswidth = 3
        else # if bounded
            mcol = :white ; mstrcol=label_antivortex[i] ; mswidth = 3
        end
        scatter!(correct_location(list_antivortex[i],L).+(0.5,0.5),m=:circle,color=mcol,msc=mstrcol,msw=mswidth,ms=:10,label=nothing)
    end
    display(p)
    # savefig("figures\\Vizu_vortices_8.pdf")
""
## Number of free vortices for the XY Model
using Plots,Statistics,Distributed,Distributions,ColorSchemes
pyplot(box=true,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3) ; plot()
include("methods.jl");

Var = 0.0 ; Ts = Array(vcat(0.0:0.1:0.7,0.725:0.025:1.2)) ; L = 200 ; init = "LowTemp" ; tmax = 500 ; R = 30
threshold_dists = [1,1.5,3,4,5,7]

Var = 0.0 ; Ts = Array(vcat(0.0:0.1:1.2)) ; L = 100 ; init = "LowTemp" ; tmax = 500 ; R = 12
threshold_dists = [3]
number_all_vortex = fill(-1,length(Ts),length(threshold_dists),R)
number_free_vortex = fill(-1,length(Ts),length(threshold_dists),R)
z = @elapsed Threads.@threads for r in 1:R
    println("r = $r on thread #$(Threads.threadid())")
    for ii in eachindex(Ts)
        T = Ts[ii]
        # println("T = $T")

        lattice,dt = create_Lattice(L,T,Var,init)
        dt *= 2

        t = 0.0
        while t < tmax
            lattice = update(lattice,T,L,dt)
            t = t + dt
        end
        # number_all_vortex[i] = length(spot_vortices(lattice[:,:,1]))
        # ~,~,number_free_vortex[i] = classify_vortices(lattice[:,:,1])

        list_vortex,list_antivortex  = spot_vortex_antivortex(lattice[:,:,1]) # Vector{Tuple{Int,Int}}
        m = length(list_vortex) # number of pairs vortex/antivortex

        label_vortex = fill(-1,m,length(threshold_dists)) ; label_antivortex = fill(-1,m,length(threshold_dists))

        dist_matrix = Array{Float64,2}(undef,m,m)
        for j in 1:m # j is an antivortex
            for i in 1:m # i is a vortex
                dist_matrix[i,j] = dist(list_vortex[i],list_antivortex[j],L)
            end
        end


        try # if there is no vortex at all, hungarian rises "ArgumentError: reducing over an empty collection is not allowed"
            matching,~ = Hungarian.hungarian(dist_matrix) # here is done the optimal bipartite matching
            for tt in eachindex(threshold_dists)
                threshold_dist = max(threshold_dists[tt],L/m)
                for i in 1:m # i is a vortex
                    j = matching[i] # j is an antivortex
                    if dist_matrix[i,j] ≤ threshold_dist # you just found its pair, so label it with a unique token !
                        label_vortex[i,tt] = 1
                        label_antivortex[j,tt] = 1
                    else # its closest pair is further than the threshold -> free vortex
                        label_vortex[i,tt] = label_antivortex[j,tt] = 0
                    end
                end
                global number_free_vortex[ii,tt,r] = 2sum(label_vortex[:,tt] .== 0)
            end # ends for loop
                global number_all_vortex[ii,:,r]  = 2m*ones(length(threshold_dists))
            catch e
                println("Caught Error ",sprint(showerror, e))
                # if there was no vortex at all in the first place, output this
                global number_free_vortex[ii,:,r] = zeros(length(threshold_dists))
                global number_all_vortex[ii,:,r]  = zeros(length(threshold_dists))
        end # ends try/catch
    end # end for Temperature loop
end # ends realisations loop

# JLD.save("data/vortex_threshold.jld","L",L,"Var",Var,"Ts",Ts,"init",init,"tmax",tmax,"thresholds",threshold_dists,"number_all_vortex",number_all_vortex,"number_free_vortex",number_free_vortex,"R",R,"tCPU[s]",14000)
L,Var,Ts,init,tmax,thresholds,number_all_vortex,number_free_vortex,R,tCPU = JLD.load("data/vortex_threshold.jld","L","Var","Ts","init","tmax","thresholds","number_all_vortex","number_free_vortex","R","tCPU[s]")

plin = plot(xlabel=L"T",ylabel=L"n/L^2",size=(450,400))
    plot!(Ts,mean(number_all_vortex,dims=3)[:,1,1]/L^2,label="All vortices",m=:utriangle,c=:black)
    for tt in eachindex(thresholds)
        plot!(Ts,mean(number_free_vortex,dims=3)[:,tt,1]/L^2,c=tt,m=:circle,label="c=$(thresholds[tt])")
    end
    xticks!([0,0.3,0.6,0.9,1.2],string.([0,0.3,0.6,0.9,1.2]))
    annotate!(0.07,0.008,text("(a)",:black,:center,15))

plog = plot(xlabel=L"T",ylabel=L"n + 1",yaxis=:log,size=(450,400),legend=nothing)
    plot!(Ts,((mean(number_all_vortex,dims=3)[:,1,1] .+ 1)),label="All vortices",m=:utriangle,c=:black)
    for tt in eachindex(thresholds)
        plot!(Ts,(mean(number_free_vortex,dims=3)[:,tt,1] .+ 1),c=tt,m=:circle,label="c=$(thresholds[tt])")
    end
    xticks!([0,0.3,0.6,0.9,1.2],string.([0,0.3,0.6,0.9,1.2]))
    annotate!(0.07,2,text("(b)",:black,:center,15))

p = plot(plin,plog,layout = (2,1),size=(450,800))
savefig("figures_draft\\threshold_vortices.pdf")

p = plot(Ts,mean(number_all_vortex,dims=3)[:,1,1],m=:o,label="All Vortices",c=:black)
for tt in eachindex(threshold_dists)
    plot!(Ts,mean(number_free_vortex,dims=3)[:,tt,1],c=tt,m=:o,label="Free Vortices, Threshold = max($(threshold_dists[tt]),L/m)")
end
xlabel!("T")
ylabel!(L"n_v")
title!("L = $L , IC = $init , tmax = $tmax")
display(p)
# savefig("figures\\vortices\\threshold_vortices_lin.pdf")

## Densité de vortex
Var = 0.5 ; T = 0 ; L = 100 ; init = "HighTemp" ; transients = 400 ; tau = 100 ; tmax = 2000 ; nn = 26 ; R = 6
distances = Float64[]
aimed = Int(5E4)
start_time = now()
@time while length(distances) < aimed
    Threads.@threads for r in 1:R
        lattice,dt = create_Lattice(L,T,Var,init)
        t = 0.0
        while t < transients
            lattice=update(lattice,T,L,dt)
            t = t + dt
        end
        for i in 1:nn
            while t < transients + i*tau
                lattice=update(lattice,T,L,dt)
                t = t + dt
            end
            list_vortex,list_antivortex = spot_vortex_antivortex(lattice[:,:,1])
            for vortex in list_vortex
                for antivortex in list_antivortex
                    d = dist(vortex,antivortex,L)
                    if d≤10 push!(distances,d) end
                end
            end
            # println("r = $r : $i/$nn")
        end # for i
    end # for r
    curr_time = now()
    dtime = curr_time - start_time
    work_done = length(distances)/aimed
    println("There are currently $(length(distances)) vortices.")
    if  0 < work_done < 1  println(" Approx time left : $(dtime.value*(1-work_done)/work_done/1000/60) minutes. ") end
end # while loop
# using JLD,HDF5
f = jldopen("data\\gr_vortex.jld", "r+")
write(f,"SRK/dataVar$Var/T",T )
write(f,"SRK/dataVar$Var/IC",init)
write(f,"SRK/dataVar$Var/distances",distances)
write(f,"SRK/dataVar$Var/nb",aimed)
write(f,"SRK/dataVar$Var/Var",Var)
write(f,"SRK/dataVar$Var/L",L)
close(f)

# delete!(f,"new_variable")
load("data\\gr_vortex.jld")
# histogram(distances,normalize=:pdf,label="T=$T , σ²=$Var",xlabel="r",ylabel="P (∃ a vortex at distance r)")
dXY = filter(x->x≤10,load("data/densities_vortices.jld")["distancesXY"])
dDeXY = filter(x->x≤10,load("data/densities_vortices.jld")["distancesDeXY"])
dSRK = filter(x->x≤10,load("data/densities_vortices.jld")["distancesSRK"])

# distances = dSRK
Var,distances,IC,T,L = values(load("data\\gr_vortex.jld")["XY"]["dataT1.1"])
# distancesN = distances + rand(Normal(0,0.05^2),length(distances))
using StatsBase,LinearAlgebra
h = fit(Histogram,distances,0:0.1:floor(Int,maximum(distances)))
# h = normalize(h,mode=:density) # if bins not equal
r = Array(h.edges[1])
dr = r[2]-r[1]
g = h.weights/π/dr ./ r[1:end-1]
g_gaz = length(distances)/π/maximum(distances)^2
# plot(xlabel="r",ylabel="g(r)",r[1:end-1],fill(1,length(r[1:end-1])),xlims=(0,10),c=:grey,line=:dash,label=nothing)
plot!(r[1:end-1],g/g_gaz,rib=0,label="T=$T , σ²=$Var , IC=$init")
lens!([6,10],[0,10],inset=(1,bbox(0.3,0.25,0.4,0.4)),label=false)



# savefig("figures\\vortices\\gr.pdf")
# JLD.save("data/densities_vortices.jld","distancesXY",distancesXY,"distancesSRK",distancesSRK,"distancesDeXY",distancesDeXY,"init",init,"transients",transients,"tau",tau,"tmax",tmax)
# savefig("figures\\vortex_density_shape.pdf")

## Critical Variance wrt #free vortices
Ls = unique(round.(Int,10.0 .^ range(log10(40),log10(300),length=15)))
R = 6 ; init = "LowTemp" ; Tkt = 0.89 ; T=0.2
tmax = 1500 # 2000 for SRK
Var_c = fill(NaN,length(Ls),R)
@time Threads.@threads for r in 1:R
    for i in eachindex(Ls)
        L = Ls[i]
        guess = (Tkt-T)/Tkt*3/log(L)^2# for T = 0.4
        # guess = 4/log(L)^2 # for T = 0
        Vars = Array(round(guess,digits=2):0.02:2)
        for Var in Vars
            lattice,dt = create_Lattice(L,T,Var,init)
            t = 0.0
            while t < 0.5*tmax lattice = update(lattice,T,L,dt) ; t += dt end
            ~,~,~,free_vortices = classify_vortices(lattice[:,:,1])
            if free_vortices > 0
                println("r = $r , L = $L , critic. σ² = $Var, departure at from $(Vars[1]) (stopped early)")
                Var_c[i,r] = Var
                break # start again with another L
            end
            while t < tmax lattice = update(lattice,T,L,dt) ; t += dt end
            ~,~,~,free_vortices = classify_vortices(lattice[:,:,1])
            if free_vortices > 0
                println("r = $r , L = $L , critic. σ² = $Var, departure at from $(Vars[1])")
                Var_c[i,r] = Var
                break # start again with another L
            end
        end
    end
end
# plot(1 ./log.(Ls).^2,minimum(Var_c,dims=2))
# plot!(1 ./log.(Ls).^2,(Tkt-T)/Tkt*(3 ./log.(Ls).^2))


plot(xlabel="1/log² L",ylabel=L"\sigma^2_c")
llog2 = log.([40,Int(1E12)]).^2
col = 1
data = JLD.load("data/critical_variance_vs_L_T0.jld")
Tkt = 0.89 ; s = round(6((Tkt - data["T"])/Tkt)^2,digits=1)
plot!(1 ./log.(data["Ls"][8:end]).^2,(mean(data["Var_c"],dims=2)[8:end]),rib=(std(data["Var_c"],dims=2)[8:end]),c=col,m=:circle,label="T = $(data["T"]) ; Slope = 6"*L"\left(\frac{T_{KT}-T}{T_{KT}}\right)^2"*"= $s")
plot!(1 ./llog2,s ./llog2 ,c=col,line=:dash,label=nothing)

col = 2
data = JLD.load("data/critical_variance_vs_L_T0.2.jld")
Tkt = 0.89 ; s = round(6((Tkt - data["T"])/Tkt)^2,digits=1)
plot!(1 ./log.(data["Ls"]).^2,(mean(data["Var_c"],dims=2)),rib=(std(data["Var_c"],dims=2)),c=col,m=:circle,label="T = $(data["T"]) ; Slope = 6"*L"\left(\frac{T_{KT}-T}{T_{KT}}\right)^2"*" = $s")
plot!(1 ./llog2,s ./llog2 ,c=col,line=:dash,label=nothing)


col = 3
data = JLD.load("data/critical_variance_vs_L_T0.4.jld")
Tkt = 0.89 ; s = round(6((Tkt - data["T"])/Tkt)^2,digits=1)
plot!(1 ./log.(data["Ls"]).^2,(mean(data["Var_c"],dims=2)),rib=(std(data["Var_c"],dims=2)),c=col,m=:circle,label="T = $(data["T"]) ; Slope = 6"*L"\left(\frac{T_{KT}-T}{T_{KT}}\right)^2"*" = $s")
plot!(1 ./llog2,s ./llog2 ,c=col,line=:dash,label=nothing)

plot!(1 ./llog2, 1/0.44^2 ./llog2,c=:grey,line=:dash,label="Slope = 5.16 from Lee et al 2010")
p2 = twiny()
plot!(p2,1 ./log.(data["Ls"]).^2,NaN*data["Ls"],label=nothing,xlabel="L",yticks=nothing,xticks=(1 ./log.([40,50,100,200,300,1000]).^2,string.([40,50,100,200,300,1000])))

ylims!((0,0.5))
xlims!((0,.08))
# savefig("figures/critical_variance_vs_L.pdf")
# savefig("figures/critical_variance_vs_L.svg")
# JLD.save("data/critical_variance_vs_L_T0.2.jld","Var_c",Var_c,"Ls",Ls,"tmax",tmax,"T",T,"init",init,"R",R,"tCPU",34000)

## ??
L = 100 ; init = "HighTemp"; T = 0.1; Var = 0.1; R = 270; tmax = 1000
# nsave = 10 ; tsave = unique(round.(Int,2.0 .^ range(log2(100),log2(tmax),length=nsave))) ; nsave = length(tsave)
deltas = fill(-1.0,2,R)
@time Threads.@threads for r in 1:R
    println("r = $r")
    t = 0.0 ;
    lattice,dt = create_Lattice(L,T,Var,init)
    while t < tmax lattice = update(lattice,T,L,dt) ; t += dt end
    global deltas[:,r] = get_Delta(lattice[:,:,1])
end
histogram(deltas[1,:],normalize=:pdf)



## Numerical Integration for the Proba of runaways
using QuadGK
ds = 0.01 #discretistion of the sigma axis
Vars = Array(ds:ds:0.4)
seuils = Array(0.6:0.2:2)
probs = zeros(5,length(Vars),length(seuils))
    for j in eachindex(seuils)
        for i in eachindex(Vars)
            seuil = seuils[j]
            σ = sqrt.(Vars[i])
            function f4c(w)
                p = 1/2*(erf((seuil-w)/σ/sqrt(2)) + erf((seuil+w)/σ/sqrt(2)))
                return p^4 * exp(-w^2/(2σ^2)) / sqrt(2pi*σ^2)   # 4 conducting links
            end
            function f3c(w)
                p = 1/2*(erf((seuil-w)/σ/sqrt(2)) + erf((seuil+w)/σ/sqrt(2)))
                return 4(1-p)*p^3 * exp(-w^2/(2σ^2)) / sqrt(2pi*σ^2)  # 3 conducting links
            end
            function f2c(w)
                p = 1/2*(erf((seuil-w)/σ/sqrt(2)) + erf((seuil+w)/σ/sqrt(2)))
                return 6(1-p)^2*p^2 * exp(-w^2/(2σ^2)) / sqrt(2pi*σ^2)  # 2 conducting links
            end
            function f1c(w)
                p = 1/2*(erf((seuil-w)/σ/sqrt(2)) + erf((seuil+w)/σ/sqrt(2)))
                return 4(1-p)^3*p * exp(-w^2/(2σ^2)) / sqrt(2pi*σ^2)  # 1 conducting links
            end
            function f0c(w)
                p = 1/2*(erf((seuil-w)/σ/sqrt(2)) + erf((seuil+w)/σ/sqrt(2)))
                return (1-p)^4 * exp(-w^2/(2σ^2)) / sqrt(2pi*σ^2)  # 0 conducting links
            end
            probs[5,i,j] = quadgk(f0c,-Inf,Inf)[1]
            probs[4,i,j] = quadgk(f4c,-Inf,Inf)[1]
            probs[3,i,j] = quadgk(f3c,-Inf,Inf)[1]
            probs[2,i,j] = quadgk(f2c,-Inf,Inf)[1]
            probs[1,i,j] = quadgk(f1c,-Inf,Inf)[1]
        end


        # x = (stds/seuil)
        # P = probs[5,:]*1 + probs[1,:]*3/4 + probs[2,:]*1/2 + probs[3,:]*1/4
        # plot!((stds/seuil) .^2,P,label="ν = $seuil",rib=0)
        # plot!((stds/seuil) .^2,1 .+erf.(-0.5(stds/seuil).^1))
        # plot!(x .^2,(1 ).*exp.(- x))
    end

P  = probs[1,:,:]*1/4 + probs[2,:,:]*1/2 + probs[3,:,:]*3/4 + probs[4,:,:]
# p1 = plot(xlabel=L"(σ/ν)²",ylabel=L"\mathcal{P}",legend=:bottomleft)
#     for j in eachindex(seuils)
#         plot!(Vars/seuils[j].^2,P[:,j],c=j,lw=3) ; plot!([NaN,NaN],rib=0,c=j,label="ν = $(seuils[j])")
#     end
#     # plot!((stds/minimum(seuils)) .^2,exp.(-(stds/minimum(seuils)).^1),c=:black,label="exp(-σ/ν)")
#     # plot!(0:0.01:1,erf.(1 ./2sqrt.(0:0.01:1)),c=:black,line=:dash,label=L"erf(\nu/2\sigma)")
#     xaxis!((0,1))
#     yaxis!((0.5,1.02))
# savefig("figures\\Pa.svg")
using SpecialFunctions
xx = 0.1:0.001:0.6
-log.(erf.(1 ./(2xx)))
p = plot()
    for ν in [1]
        plot!(xx,-1 ./ log.(erf.(2*0.222 ./(2xx))),label="ν = $ν",axis=:log)
        # plot!(xx,-1 ./ log.(erf.(2*2/pi ./(2xx))),label="ν = $ν",line=:dash,axis=:log)
        # plot!(xx,-1 ./ log.(erf.(2*1/pi ./(2xx))),label="ν = $ν",line=:dash,axis=:log)
    end
    plot!(xx , 1 ./ xx,axis=:log,c=:black)
    xlabel!(L"σ")

ylabel!(L"-\log\ \mathcal{P}")
    plot!(c=:grey,[0.2,0.2],[0,0.35],line=:dash)
    annotate!(0.2,0.40, text("Linear approx. holds ",:center,12))
    annotate!(0.2,0.37, text("at from "*L"σ² \approx 0.04",:center,12))
    annotate!(0.33,0.25, text(L"-\log\,\mathcal{P}\sim σ",:center,12,45.0))
    # p2 = twiny()
    # plot!(p2,sqrt.(0.01:0.001:0.2),NaN*sqrt.(0.01:0.001:0.2),xlabel=L"σ^2",xticks=(Array(0.01:0.01:0.25).^0.5,[0.01,0.02,0.03,0.04,0.05,"","","","",0.1,"","","","","","","","","",0.2]))
savefig("figures\\P.pdf")

L = 200
    links = Float64[]
    thetas = mod.(lattice[:,:,1],2pi)
    # thetas = 2pi*rand(L,L) - 2pi*rand(L,L)
    # thetas = 2pi*rand(L,L)
    display(histogram(vec(thetas),normalize=true))
    for i in 1:L
        for j in 1:L
            theta = thetas[i,j]
            n = get_angle_nearest_neighbours(thetas,i,j,L)
            for el in n
                push!(links,abs(sin(theta-el)))
            end
        end
    end
    mean(links)


# L = 500 ; Vars = 0.0:0.025:1.0
# r4 = zeros(length(Vars))
# r3 = zeros(length(Vars))
# r2 = zeros(length(Vars))
# r1 = zeros(length(Vars))
# r0 = zeros(length(Vars))
# for i in eachindex(Vars)
#     lat,dt = create_Lattice(L,0.0,Vars[i],"HighTemp")
#     r4[i] = length(spot_runaways(lat[:,:,2],2,4))/L^2
#     r3[i] = length(spot_runaways(lat[:,:,2],2,3))/L^2
#     r2[i] = length(spot_runaways(lat[:,:,2],2,2))/L^2
#     r1[i] = length(spot_runaways(lat[:,:,2],2,1))/L^2
#     r0[i] = length(spot_runaways(lat[:,:,2],2,0))/L^2
# end
# plot(xlabel="σ²",ylabel=L"p_n")
# plot!(Vars,r4,ribbon=0,m=:o,line=nothing)
# plot!(Vars,r3,ribbon=0,m=:o,line=nothing)
# plot!(Vars,r2,ribbon=0,m=:o,line=nothing)
# plot!(Vars,r1,ribbon=0,m=:o,line=nothing)
#
# plot!(stds .^2,probs[4,:],c=1)
# plot!(stds .^2,probs[3,:],c=2)
# plot!(stds .^2,probs[2,:],c=3)
# plot!(stds .^2,probs[1,:],c=4)
#
# plot!((NaN,NaN),ribbon=0,label=L"p_3",c=4)
# plot!((NaN,NaN),ribbon=0,label=L"p_2",c=3)
# plot!((NaN,NaN),ribbon=0,label=L"p_1",c=2)
# plot!((NaN,NaN),ribbon=0,label=L"p_0",c=1)
# scatter!((NaN,NaN),m=:circle,c=:grey,label="Data")
# plot!((NaN,NaN),c=:grey,label="Predictions")
# # savefig("figures\\pn.pdf")

## Parisi Overlap
# L = 100
# Ts  = [0.1,0.3,0.5,0.7]
# Vars = [0.01,0.02,0.05]
# R = 6 #
# tmax = 1000
# init = "HighTemp"
#
# q = Array{Number}(undef,(2,length(Ts),length(Vars),L,L,R))
#
# @time Threads.@threads for r in 1:R
#     for i in eachindex(Ts)
#         for j in eachindex(Vars)
#             T = Ts[i]
#             Var = Vars[j]
#             lattice1,dt = create_Lattice(L,T,Var,init)
#             lattice2 = lattice1
#             t = 0.0
#             while t<tmax
#                 lattice1 = update(lattice1,T,L,dt)
#                 lattice2 = update(lattice2,T,L,dt)
#                 t += dt
#             end
#             q[1,i,j,:,:,r] = cos.(lattice1[:,:,1] - lattice2[:,:,1])
#             q[2,i,j,:,:,r] = cos.(lattice1[:,:,1]) .* cos.(lattice2[:,:,1])
#         end
#     end
# end
# using LinearAlgebra,StatsBase
# p = plot(xlabel="q",ylabel="P(q)")
# for i in eachindex(Ts)
#     for j in eachindex(Vars)
#         x = vec(q[2,i,j,:,:,:])
#         h = fit(Histogram,x,minimum(x):0.01:maximum(x))
#         h = normalize(h,mode=:density)
#         plot!(Array(h.edges[1])[1:end-1],h.weights/length(x),yaxis=:log,rib=0,label="T = $(Ts[i]) , σ² = $(Vars[j])")
#     end
# end
# xlims!((-1.1,1.1))
# display(p)
# # savefig("figures\\Pq_XY_closeTk.pdf")
#
#
# x = vec(q[2,4,3,:,:,:])
# h = fit(Histogram,x,minimum(x):0.01:maximum(x))
# h = normalize(h,mode=:density)
# plot(Array(h.edges[1])[1:end-1],h.weights/length(x),rib=0)


## Energy considerations
L = 100 ; init = "LowTemp"; T = 0.; Var = 0.15; tmax = 2000
    lattice,dt = create_Lattice(L,T,Var,init)
    t = 0.0
    energies = []
    @time while t < tmax
        global lattice = update(lattice,T,L,dt)
        global t += dt
        ~,tmp = energy(lattice[:,:,1])
        push!(energies,tmp)
    end
plot((energies)/energies[1],xaxis=:log)
energy_loc,~ = energy(lattice[:,:,1])
minn = abs(minimum(energy_loc))
heatmap(energy_loc'.+minn)
frontiers = zeros(size(energy_loc))
    for i in eachindex(energy_loc)
        if 0.29 < energy_loc[i] + minn < 0.7 frontiers[i] = 1 end
    end
    heatmap(frontiers,c=cgrad([:white,:black]))
savefig("figures\\energy_picture.pdf")
## Transmission through chain of spins
Vars = Array(0.01:0.01:0.3)
seuils = Array(0.6:0.2:2)
N = 100
R = Int(1E4)
B     = zeros(N-1,R,length(Vars),length(seuils))
# Bprod = zeros(N-1,R,length(Vars),length(seuils))

@time for l in eachindex(Vars)
        for s in eachindex(seuils)
            seuil = seuils[s]
            Var = Vars[l]
            omegas = rand(Normal(0,sqrt(Var)),N,R)
            for j in 1:R
                for i in 1:N-1
                    global B[i,j,l,s] =  abs(omegas[i,j] - omegas[i+1,j]) < seuil
                end
            end
            # for j in 1:R
            #     for i in 1:N-1
            #         global Bprod[i,j,l,s] = prod(B[1:i,j,l,s])
            #     end
            # end
            # plot!(mean(Bprod[:,:,l],dims=2),rib=0std(Bprod,dims=2),yaxis=:log,m=*,label="σ² = $Var")
        end
    end
B

Bmean = zeros(length(Vars),length(seuils))
Bstd  = zeros(length(Vars),length(seuils))
for s in 1:length(seuils)
    for i in 1:length(Vars)
        global Bmean[i,s] = mean(B[:,:,i,s])
        global Bstd[i,s]  = std(B[:,:,i,s])
    end
end
ms = minimum(seuils)^2
p=plot()
    for s in eachindex(seuils)
        plot!(Vars./seuils[s]^2,Bmean[:,s],c=s,rib=0Bstd[:,s],label="ν = $(seuils[s])")
    end
    plot!(Vars./ms,erf.(1 ./2sqrt.(Vars./ms)))
    display(p)


p=plot()
for l in eachindex(Vars)
    plot!(mean(Bprod[:,:,l],dims=2),rib=0std(Bprod[:,:,l],dims=2),yaxis=:log,label="σ² = $(Vars[l])")
end
display(p)


## Different representations
T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_Z.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
smoothedE = (energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
debut = 5780
fin   = 7100
traj = Matrix{Tuple{Int64,Int64}}(undef,2,fin-debut+1)
for i in debut:fin
    a,b = spot_vortex_antivortex(thetas_hist[floor(Int,i/every)])
    traj[:,i-debut+1] = [a[1].-(44,44),b[1].-(44,44)]
end
p2d = plot(size=(600,600),Array(3100:every:9900)*dt.+ttr,smooth(smoothedE[floor(Int,3100/every)+1:floor(Int,9900/every)+1]),xlims=(450,n*dt+ttr),xticks=([500,750,1000,1250],[500,750,1000,1250]))
scatter!((ttr+dt*debut,(smoothedE[floor(Int,debut/every)])),ms=8,m=:circle,c=:black)
scatter!((ttr+dt*fin,(smoothedE[floor(Int,fin/every)])),ms=8,m=:circle,c=:black)
ylims!((305,345))
# ylabel!(L"\sum_i\ E_i(t)")
annotate!(530,308.5,text("(d)",:black,:center,15))
annotate!(1500,308.5,text("t",:black,:center,15))
annotate!(960,337,text(L"t_0",:black,:left,15))
annotate!(1165,308.5,text(L"t_1",:black,:left,15))
annotate!(480,343,text(L"\sum_i\ E_i(t)",:black,:left,15))

heatmap(size=(300,300),transpose(diffops(psi(thetas_hist[floor(Int,debut/every)+1]))[1]),clims=(0,1),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
energyy_1 = energy(thetas_hist[floor(Int,debut/every)+1])[1] .+ 4 # E_0 = -4
p2a = heatmap(size=(300,300),energyy_1',clims=(0,0.5),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
annotate!(6,5,text("(a)",:white,:center,15))
annotate!(36.5,47,text(L"t=t_0",:white,:left,15))
plot!((traj[1,:]),c=:white)
xlims!((1,50))
ylims!((1,50))
xlabel!("dummy",xguidefont = font(:black))

energyy_2 = energy(thetas_hist[floor(Int,fin/every)+1])[1] .+ 4
p2b = heatmap(size=(300,300),energyy_2',clims=(0,0.5),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
annotate!(6,5,text("(b)",:white,:center,15))
annotate!(36,47,text(L"t=t_1",:white,:left,15))
plot!((traj[1,:]),c=:white)
xlims!((1,50))
ylims!((1,50))
xlabel!("dummy",xguidefont = font(:black))

energyy_3 = energy(thetas_hist[end])[1] .+ 4
ppp = heatmap(size=(300,300),energyy_3',clims=(0,0.5),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")

plot(p2a,p2b,ppp,layout=(1,3),size=(900,300))
