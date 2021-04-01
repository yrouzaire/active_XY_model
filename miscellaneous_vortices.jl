"Copyright (c) 2021 Y.Rouzaire All Rights Reserved."
using Distributions,Statistics,JLD,SpecialFunctions,LsqFit
cd("D:/Documents/Ecole/EPFL/Master_Project/vortices_study")
include("methods_vortices.jl");
using Plots,LaTeXStrings,ColorSchemes
# pyplot(box=true,label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3) ; plot()
gr(box=true,label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3) ; plot()
# JLD.save("../vortices_study/data_vortices/locations_single_default.jld","L",L,"R",R,"Ts",Ts,"Vars",Vars,"tmax",tmax,"save_every",save_every,"locations",locations)
L,R,Ts,Vars,tmax,save_every,locations_single_default = JLD.load("data_vortices/locations_single_default.jld","L","R","Ts","Vars","tmax","save_every","locations")
L,R,Ts,Vars,tmax,save_every,locations_pair_defaults  = JLD.load("data_vortices/locations_pair_defaults.jld","L","R","Ts","Vars","tmax","save_every","locations")
nT = length(Ts) ; nVar = length(Vars) ; dt    = determine_dt(maximum(Ts),maximum(Vars))
Base.summarysize(locations_pair_defaults)/1E6 # size in Mo

## Parameters
L         = 150
    Q     = 1        # for a single vortex
    r0s   = Int.([10,20,30].+2) ; nR0 = length(r0s) # for two vortices
    R     = 1
    Ts    = [0.2] ; nT = length(Ts)
    Vars  = [0.0] ; nVar = length(Vars)
    TV = [(0.1,0.0),(0.2,0.0),(0.05,0.05),(0.1,0.1),(0.0,0.1)]
    tmax  = Int(3E3) ; save_every = 20
    dt    = determine_dt(maximum(Ts),maximum(Vars))

## Single Vortex
# locations_single_vortex = Array{Union{Tuple{Int,Int},Missing}}(undef,(length(0:save_every:tmax/dt),nT,nVar,R))
# @time Threads.@threads for r in 1:R
#     for i in eachindex(Ts)
#         for j in eachindex(Vars)
#             T = Ts[i] ; Var = Vars[j]
#             println("r = $r , T = $(Float16(T)) , σ² = $(Float16(Var))")
#             thetas = create_isolated_vortex(L,L,+1)
#             omegas = rand(Normal(0,sqrt(Var)),L,L)
#
#             t = 0.0
#             locations_single_vortex[:,i,j,r] = evolve_FBC(Q,thetas,omegas,T,dt,tmax,save_every)
#         end
#     end
# end
# JLD.save("../vortices_study/data_vortices/locations_single_default.jld","L",L,"R",R,"Ts",Ts,"Vars",Vars,"tmax",tmax,"save_every",save_every,"locations",locations_single_default)

## Pair of Vortices
locations_pair_vortices = Array{Union{Tuple{Int16,Int16},Missing}}(undef,(2,length(0:save_every:tmax/dt),length(TV),nR0,R))
distance_pair_vortices  = Array{Union{Float64,Missing}}(undef,(length(0:save_every:tmax/dt),length(TV),nR0,R))
@time Threads.@threads for r in 1:R
    for k in eachindex(r0s)
        for i in eachindex(TV)
            T,Var = TV[i]; r0 = r0s[k]
            println("r0 = $r0 , T = $(Float16(T)) , σ² = $(Float16(Var)) , r = $r")
            thetas = create_pair_vortices(L,r0)
            omegas = rand(Normal(0,sqrt(Var)),L,L)

            t = 0.0
            locations_pair_vortices[:,:,i,k,r],distance_pair_vortices[:,i,k,r] = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)
        end
    end
end
# JLD.save("../vortices_study/data_vortices/pair_defaults.jld","L",L,"R",R,"r0s",r0s,"TV",TV,"tmax",tmax,"save_every",save_every,"locations",locations_pair_vortices,"distance",distance_pair_vortices)

# ## 2D Histograms Single
# for j in 1 # Vars
#     for i in 1:length(Ts) # Ts
#         x = [] ; y = [] ;
#         for element in vec(locations[:,i,j,:])
#             if !ismissing(element)
#                 push!(x,element[1])
#                 push!(y,element[2])
#             end
#         end
#         h=histogram2d(x,y, show_empty_bins = false, normed = true,lims=(0,L),c=cgrad([:blue,:green,:orange,:red]),title="L = $L, T = $(Ts[i]), σ² = $(Vars[j])")
#         display(h)
#         # savefig("../vortices_study/figures_vortices/hist2D_DeXY_T$(Ts[i])_Var$(Vars[j]).pdf")
#         # savefig("../vortices_study/figures_vortices/hist2D_XY_T$(Ts[i]).pdf")
#     end
# end

## Distance
distance_pair_vortices_avg  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0)) ; distance_pair_vortices_std  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0))
for n in 1:size(distance_pair_vortices)[3]
    for i in 1:size(distance_pair_vortices)[2]
        for t in 1:size(distance_pair_vortices)[1]
            distance_pair_vortices_avg[t,i,n] = mean(skipmissing(distance_pair_vortices[t,i,n,:]))
            distance_pair_vortices_std[t,i,n] = std(skipmissing(distance_pair_vortices[t,i,n,:]))
        end
    end
end

tv = 1
    tt = Array(0:dt*save_every:tmax)
    p0 = [0.02]
    lastindex = [60,223,404]
    col = [:blue,:orangered1,:green]
    p  = plot(xlabel="t",ylabel="⟨r(t)⟩",legend=:best)
    for n in 1:nR0
        r0 = r0s[n].-2
        plot!(tt,distance_pair_vortices_avg[:,1,n],rib=distance_pair_vortices_std[:,1,n],label=L"r_0 = "*string(r0s[n]-2),c=n)
        @. model(x, p) = r0*exp.(- (erfinv.(p[1]*x/r0)).^2)
        ind = lastindex[n]
        fitt = curve_fit(model, tt[1:ind], distance_pair_vortices_avg[1:ind,1,n], p0)
        plot!(tt[1:ind],model(tt[1:ind],coef(fitt)),c=:black)
        plot!(tt,distance_pair_vortices_avg[:,5,n],line=:dash,c=col[n])
        end
    xlims!((-40,2000))
    ylims!((-10,33))
    plot!([NaN,NaN],[NaN,NaN],c=:black,label="Trajectory with log(r) potential")
    plot!([NaN,NaN],[NaN,NaN],c=:grey,label="(solid lines) T = 0.1 , σ² = 0.0 ")
    plot!([NaN,NaN],[NaN,NaN],line=:dash,c=:grey,label="(dash lines) T = 0.0 , σ² = 0.1 ")
savefig("figures\\vortices\\rt.pdf")
# savefig("../vortices_study\\figures_vortices\\rt.svg")
# Fitting
# green
r0 = 10
ind = 60
# tt[ind]
coef(fitt)
plot!(tt[1:ind],model(tt[1:ind],coef(fitt)))



## Mean Square Displacement
sum(ismissing.((locations_single_default)))
loc = convert(Array{Union{Tuple{Int16,Int16},Missing}},locations_single_vortex)
MSD_avg,MSD_std = SD(L,nT,nVar,R,loc)
p=plot(xlabel="t",ylabel=L"⟨(r(t)-r_0)^2⟩")
    couleur = 1
    styles = [:solid,:dashdot,:dash]
    t = Array(0:save_every:tmax/dt)*dt
    tlogt = t ./ log.(t)
    for j in 1:nVar # Vars
        for i in 1:nT # Ts
            plot!(t[2:end],remove_negative(MSD_avg[2:end,i,j],1/R),c=couleur,rib=0,label="T = $(Ts[i]) , σ² = $(Vars[j])",axis=:log)
            global couleur += 1
        end
    end
    display(p)
    # remove_negative(MSD_avg[2:end,1,1],1/R)
    plot!(t[55:end],0.06t[55:end].^1.5,c=:black,axis=:log,label=L"t^{3/2}")
    plot!(t[30:end],0.1tlogt[30:end],c=:black,axis=:log,label="t/log t",line=:dash)
    # savefig("../vortices_study/figures_vortices/MSD.pdf")
# savefig("../vortices_study/figures_vortices/MSD_DeXY.pdf")
# savefig("../vortices_study/figures_vortices/MSD_XY.pdf")

## Stiffness
heatmap(stiffness(locations_single_default),c=cgrad([:blue,:green,:orange,:red]))
stiffness(locations_single_default)

## GIF of trajectories [using Plots ; gr()]
function GIF(locations_single_default::Vector{Tuple{Union{Missing, Int64},Union{Missing, Int64}}})
    anim = @animate for i in eachindex(locations_single_default)
        plot(locations_single_default[1:i],m=:circle,ms=5)
        scatter!(locations_single_default[i],c=:red,ms=7)
        title!("L = $L , T = $T , σ² = $Var , t = $(round(dt*i*save_every,digits=1))")
    end
    return anim
end

L = 200 ; Q = 1 ; T = 0.0 ; Var = 0.15 ; tmax = 2000 ; dt = determine_dt(T,Var) ; save_every = 10 ; R = 6
anims = Vector{Animation}(undef,R)

# Simulate Trajectories
@time trajs = trajectory(L,T,Var,tmax,dt,Q,save_every,R)

# Make GIFs out of it
@time for r in 1:R anims[r] = GIF(trajs[:,r]) end
@time for r in 1:R gif(anims[r],"../vortices_study\\figures_vortices/gifs_trajectories\\L$(L)_T$(T)_Var$(Var)_$r.gif",fps=20) end

# All trajectories at final time on one plot
p=plot(title="L = $L , T = $T , σ² = $Var",lims=(0,L)) ; for r in 1:R plot!(trajs[:,r]) end ; display(p)
# savefig("../vortices_study\\figures_vortices\\paths_L$(L)_T$(T)_Var$(Var).pdf")

# Visualize Vector field as the same time WARNING : small times and small lattices only
L = 60 ; Q = 1 ; T = 0.1 ; Var = 0.2 ; tmax = 100 ; dt = determine_dt(T,Var) ; save_every = 5 ; R = 1
anims = Vector{Animation}(undef,R)
grille = [(i,j) for i=1:L, j=1:L]  ; xs   = [grille[n][1] for n in 1:length(grille)]  ; ys   = [grille[n][2] for n in 1:length(grille)] ;
function GIF(locations_single_default,thetas_history,omegas_history,xs::Vector{Int64},ys::Vector{Int64})
    for element in locations_single_default element = element .+ (.5,.5) end
    anim = @animate for i in eachindex(locations_single_default)
        thetas = thetas_history[i,:,:]
        heatmap(energy(thetas)',c=cgrad([:blue,:white,:red]),clims=(-1.,1.))
        quiver!(xs,ys,quiver=(vec(cos.(thetas)),vec(sin.(thetas))),label=false,color=:black,size=(1000,1000))
        scatter!(locations_single_default[i],c=:black,m=:circle,ms=10)
        plot!(locations_single_default[1:i],c=:black,lw=5)
        title!("L = $L , T = $T , σ² = $Var , t = $(round(dt*i*save_every,digits=1))")
    end
    return anim
end
@time trajs,thetas_history,omegas_history = trajectory_and_angles(L,T,Var,tmax,dt,Q,save_every,R)
@time for r in 1:R
    giff = GIF(trajs[:,r],thetas_history[:,:,:,r],omegas_history[:,:,r],xs,ys)
    anims[r] = giff
    println("Animation $r/$R done")
    gif(giff,"../vortices_study\\figures_vortices/gifs_trajectories\\test_L$(L)_T$(T)_Var$(Var)_$r.gif",fps=10)
end
# JLD.save("../vortices_study\\data_vortices\\quiver_L$(L)_T$(T)_Var$(Var)_R$R.jld","L",L,"Q",Q,"T",T,"Var",Var,"tmax",tmax,"dt",dt,"save_every",save_every,"R",R,"anims",anims,"thetas_history",thetas_history,"trajs",trajs)
# @time for r in 1:R gif(anims[r],"../vortices_study\\figures_vortices/gifs_trajectories\\quiver_L$(L)_T$(T)_Var$(Var)_$r.gif",fps=20) end


## Pour montrer des réalisations XY et NKM
L         = 100
    Q     = 1        # for a single vortex
    r0    = 30      # for a pair of vortices
    T     = 0.1
    Var   = 0.1
    tmax  = Int(3E3) ; save_every = 2
    dt    = determine_dt(maximum(T),maximum(Var))

# Pair of Vortices
thetas = create_pair_vortices(L,r0)
    omegas = rand(Normal(0,sqrt(Var)),L,L)

    @time locations,distance = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)
    locations_plot = Matrix{Tuple{Float64,Float64}}(undef,size(locations))
    for i in eachindex(locations)
        element = locations[i]
        if ismissing(element) locations_plot[i] = (NaN,NaN)
        else locations_plot[i] = Float64.(element)
        end
    end
    plot((locations_plot[1,:]),line=:solid,c=3)
    plot!((locations_plot[2,:]),line=:solid,c=4)
    xlims!((0,L)) ; ylims!((0,L))
# JLD.save("data_vortices\\trajectory_pairNKM3.jld","T",T,"Var",Var,"L",L,"tmax",tmax,"dt",dt,"r0",r0,"locations",locations_plot)


## Pour étudier la correlation Energie vortex
gr(box=true,label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3) ; plot()
using Distributed
include("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model/methods.jl")
0.2*(log(60)/log(L))^2
L         = 100
    Q     = 1       # for a single vortex
    r0    = Int(L/2)    # for a pair of vortices
    T     = 0.0
    Var   = 0.1
    ttr   = Int(00)
    dt    = determine_dt(maximum(T),maximum(Var))
    grille = [(i,j) for i=1:L, j=1:L]  ; xs   = [grille[n][1] for n in 1:length(grille)]  ; ys   = [grille[n][2] for n in 1:length(grille)] ;

    # Pair of Vortices
    thetas = create_pair_vortices(L,r0)
    # thetas = create_Lattice(L,T,Var,"LowTemp")[1][:,:,1]
    # thetas = create_isolated_vortex(L,L,Q)
    omegas = rand(Normal(0,sqrt(Var)),L,L)

    # Transients
    t = 0
    @time while t<ttr
        global thetas = update(thetas,omegas,L,T,dt)
        global t += dt
    end

    # Steady State
    n = 10000 ; every = 20 ;
    length(1:every:n)
    ~,E = energy(thetas) ; energy_hist = [E]
    thetas_hist = [thetas]
    for i in 1:every:n
        for e in 1:every
            global thetas = update(thetas,omegas,L,T,dt)
            global t += dt
        end
        ~,E = energy(thetas)
        push!(energy_hist,E)
        push!(thetas_hist,thetas)
    end

# # heatmap(mod.(thetas_hist[end]',2π),c=cgrad([:white,:blue,:green,:orange,:red,:white]),colorbar_title=L"\theta",subplot=1)
spot_vortices(thetas_hist[end])
# energyy = mean([energy(thetas_hist[end-i])[1] for i in 10])
# energyy .+= abs(minimum(energyy))
# heatmap(energyy.^1,clims=(0,3))


cd("D:\\Documents\\Ecole\\EPFL\\Master_Project\\vortices_study\\")
# T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_E.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
    smoothedE = smooth(energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
    dE = vcat(1,diff(smoothedE))
    plot(Array(1:every:n)*dt.+ttr,smoothedE,xlims=(0,n*dt+ttr))
""
@time animA = @animate for i in 5800:every:7000
    plot(layout=2,title="T = $T, Var=$Var, t = $(round(ttr+dt*i,digits=2))",size=(2000,1000))
    thetas = thetas_hist[floor(Int,i/every)+1]
    plot!(subplot=1,Array(1:every:n)*dt.+ttr,smoothedE,xlims=(0,n*dt+ttr),size=(1000,1000))
    scatter!(subplot=1,(i*dt+ttr,smoothedE[floor(Int,i/every)+1]),m=:star,c=:black)

    quiver!(xs,ys,quiver=(vec(cos.(thetas)),vec(sin.(thetas))),label=false,color=:black,subplot=2,aspect_ratio=1,size=(1000,1000))
    for vortex in spot_vortices(thetas)
        if dE[floor(Int,i/every)+1]>0 mcol = :red
        else mcol = :green
        end

        scatter!((vortex[1:2]).+(0.5,0.5),m=:circle,color=mcol,msc=mcol,msw=2,ms=10,subplot=2)
    end
end
    gif(animA,fps=10)
    # gif(animA,"figures_vortices/dE_E.gif",fps=10)
&
save("data_vortices\\energy_Z.jld","T",T,"Var",Var,"L",L,"ttr",ttr,"dt",dt,"r0",r0,"thetas_hist",thetas_hist,"energy_hist",energy_hist,"n",n,"every",every)



plot(Array(2700:every:9000)*dt.+ttr,smoothedE[floor(Int,2700/every)+1:floor(Int,9000/every)+1],xlims=(0,n*dt+ttr))


## E vs Var
L         = 50
    Q     = 1       # for a single vortex
    r0    = 30      # for a pair of vortices
    T     = 0.0
    Vars  = Array(0.0:0.05:1)
    R     = 20
    tmax  = Int(100)
    dt    = determine_dt(maximum(T),maximum(Var))
    E = zeros(length(Vars),R)

    @time for i in eachindex(Vars)
        for r in 1:R
            thetas = create_Lattice(L,T,Var,"LowTemp")[1][:,:,1]
            omegas = rand(Normal(0,sqrt(Var)),L,L)
            t = 0.0
            while t<tmax
                thetas = update(thetas,omegas,L,T,dt)
                t += dt
            end
            ~,E[i,r] = energy(thetas)
        end
    end
    E = E/L^2
E
plot(Vars,mean(E,dims=2).+abs(E[1,1]))


## Plot several trajectories that lead to <r(t)> = r0
L,R,r0s,TV,tmax,save_every,locations_pair_vortices,distance_pair_vortices = JLD.load("../vortices_study/data_vortices/pair_defaults0.jld","L","R","r0s","TV","tmax","save_every","locations","distance")
# distance_pair_vortices_avg  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0)) ; distance_pair_vortices_std  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0))
# for n in 1:size(distance_pair_vortices)[3]
#     for i in 1:size(distance_pair_vortices)[2]
#         for t in 1:size(distance_pair_vortices)[1]
#             distance_pair_vortices_avg[t,i,n] = mean(skipmissing(distance_pair_vortices[t,i,n,:]))
#             distance_pair_vortices_std[t,i,n] = std(skipmissing(distance_pair_vortices[t,i,n,:]))
#         end
#     end
# end
tt = Array(0:dt*save_every:tmax)

p=plot()
rsok_tv3 = [2,16,23,30,36,37,42,44,45,46,50,51,52,53]
tv = 3
for r in 1:50
    p=plot(tt,distance_pair_vortices[:,tv,1,r])
    display(p)
    sleep(1.5)
end

p=plot(title="Realisations for T,Var  = $(TV[3])")
for r in rsok_tv3
    plot!(tt,distance_pair_vortices[:,3,1,r])
end
display(p)
# savefig("figures\\realisation_r0.pdf")

## Distribution G(x,t)
L         = 200
    Q     = 1       # for a single vortex
    r0    = 100      # for a pair of vortices
    T     = 0.1
    Var   = 0.1
    R     = 600
    tmax  = Int(1E3)
    save_every = 50
    dt    = determine_dt(maximum(T),maximum(Var))
    # history_locations = Array{Union{Tuple{Int16,Int16},Missing}}(missing,2,1+floor(Int16,floor(Int,tmax/dt)/save_every),R)
    history_locations = Array{Tuple{Union{Int16,Missing},Union{Int16,Missing}}}(undef,2,1+floor(Int16,floor(Int,tmax/dt)/save_every),R)

@time Threads.@threads for r in 1:R
    println("$r")
    thetas = create_pair_vortices(L,r0)
    omegas = rand(Normal(0,sqrt(Var)),L,L)
    global history_locations[:,:,r],~ = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)
end
# save("data_vortices/data_Gdx2.jld","history_locations",history_locations,"T",T,"Var",Var,"tmax",tmax,"save_every",save_every,"L",L,"r0",r0)

history_locations,T,Var,tmax,save_every,L,r0 = load("data_vortices/data_Gdx2.jld","history_locations","T","Var","tmax","save_every","L","r0")
25*50*0.15
factor = 25
    distances = Array{Any,3}(undef,2,floor(Int16,floor(Int,tmax/dt)/save_every)-factor,R)
    for r in 1:R
        for i in 1:size(distances)[2]
            distances[1,i,r] = dist_missing(history_locations[1,i+factor,r],history_locations[1,i,r],L)
            distances[2,i,r] = dist_missing(history_locations[2,i+factor,r],history_locations[2,i,r],L)
        end
    end
    # Base.summarysize(distances)
    # data_plot = Float16.(filter(x->!ismissing(x),vec(distances)))
    # Base.summarysize(data_plot)

    histogram((filter!(x->!ismissing(x)&&x>0,vec(distances))),yaxis=:log,normalize=true)
    # histogram(data_plot,yaxis=:log,normalize=true)
    # sum(ismissing.(vec(distances)))

## Explanation dash on boundaries
cd("D:\\Documents\\Ecole\\EPFL\\Master_Project/vortices_study\\")

T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_Z.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
    smoothedE = (energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
    debut = 5780
    fin   = 7100
    traj = Matrix{Tuple{Int64,Int64}}(undef,2,fin-debut+1)
    for i in debut:fin
        a,b = spot_vortex_antivortex(thetas_hist[floor(Int,i/every)])
        traj[:,i-debut+1] = [a[1].-(44,44),b[1].-(44,44)]
    end
    a = plot(size=(600,300),Array(3100:every:9900)*dt.+ttr,smooth(smoothedE[floor(Int,3100/every)+1:floor(Int,9900/every)+1]),xlims=(450,n*dt+ttr),xticks=([500,750,1000,1250],[500,750,1000,1250]))
    scatter!((ttr+dt*debut,(smoothedE[floor(Int,debut/every)])),ms=8,m=:circle,c=:black)
    scatter!((ttr+dt*fin,(smoothedE[floor(Int,fin/every)])),ms=8,m=:circle,c=:black)
    ylims!((305,341))
    ylabel!(L"E(t)")
    annotate!(530,308.5,text("(a)",:black,:center,15))
    annotate!(1500,308.5,text("t",:black,:center,15))
    annotate!(960,337,text(L"t_0",:black,:left,15))
    annotate!(1165,308.5,text(L"t_1",:black,:left,15))
    # annotate!(480,343,text(L"E(t)",:black,:left,15))

    energyy_1 = energy(thetas_hist[floor(Int,debut/every)+1])[1] .+ 4 # E_0 = -4
    b = heatmap(size=(300,300),energyy_1'[45:95,45:95],clims=(0,1),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
    annotate!(6,5,text("(b)",:white,:center,15))
    annotate!(36.5,47,text(L"t=t_0",:white,:left,15))
    plot!((traj[1,:]),c=:white)
    xlims!((1,50))
    ylims!((1,50))
    xlabel!("dummy10",xguidefont = font(:black))


    energyy_2 = energy(thetas_hist[floor(Int,fin/every)+1])[1] .+ 4
    c = heatmap(size=(300,300),energyy_2'[45:95,45:95],clims=(0,1),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
    annotate!(6,5,text("(c)",:white,:center,15))
    annotate!(36,47,text(L"t=t_1",:white,:left,15))
    plot!((traj[1,:]),c=:white)
    xlims!((1,50))
    ylims!((1,50))
    xlabel!("dummy10",xguidefont = font(:black))

#     T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_Y.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
#     smoothedE = (energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
#     debut = 5200
#     fin   = 6600
#     traj = Matrix{Tuple{Int64,Int64}}(undef,2,fin-debut+1)
#     for i in debut:fin
#         a,b = spot_vortex_antivortex(thetas_hist[floor(Int,i/every)])
#         traj[:,i-debut+1] = [a[1].-(39,29),b[1].-(39,29)]
#     end
#     a2 = plot(size=(300,300),Array(3100:every:9999)*dt.+ttr,smooth(smoothedE[floor(Int,3100/every)+1:floor(Int,9999/every)+1]),xlims=(450,n*dt+ttr))
#     scatter!((ttr+dt*debut,(smoothedE[floor(Int,debut/every)])),ms=7,m=:circle,c=:black)
#     scatter!((ttr+dt*fin,(smoothedE[floor(Int,fin/every)])),ms=7,m=:circle,c=:black)
#     ylabel!(L"E(t)-E_0")
#     xlabel!(L"t")
#     annotate!(630,271,text("(A2)",:black,:center,15))
#     annotate!(870,311,text(L"t_0",:black,:left,15))
#     annotate!(1085,270.5,text(L"t_1",:black,:left,15))
#     ylims!((267,314))
#
#     energyy_1 = energy(thetas_hist[floor(Int,5200/every)+1])[1] .+ 4
#     b2 = heatmap(size=(300,300),energyy_1'[30:80,40:90],clims=(0,1.),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
#     annotate!(8,5,text("(B2)",:white,:center,15))
#     xlabel!(L"t=t_0 ",xguidefont = font(:black))
#     plot!((traj[1,:]),c=:white)
#     xlims!((1,50))
#     ylims!((1,50))
#
#     energyy_2 = energy(thetas_hist[floor(Int,6600/every)+1])[1] .+ 4
#     c2 = heatmap(size=(300,300),energyy_2'[30:80,40:90],clims=(0,1),colorbar=nothing,axis=nothing,foreground_color_subplot=colorant"white")
#     annotate!(8,5,text("(C2)",:white,:center,15))
#     xlabel!(L"t=t_1 ",xguidefont = font(:black))
#     plot!((traj[1,:]),c=:white)
#     xlims!((1,50))
#     ylims!((1,50))
# &
    # lay = @layout [
    # [a{0.5h} ; b{0.5w} c{0.5w}  ] d{0.5w}
    #  ]
    lay = @layout [ [a ; b{0.44w} c{0.56w}] d]
    p=plot(a,b,c,p32,layout=lay,size=(1200,600))
    # p=plot(a1,b1,c1,a2,b2,c2,layout=(2,3),size=(900,600))
    savefig(p,"figures_vortices\\dash_32.pdf")
    savefig(p,"figures_vortices\\dash_32.svg")
    display(p)
a
heatmap(rand(10,10),colorbar_title=L"E(x,y)",clims=(0,1),size=(600,600)) ; savefig("figures_vortices\\colorbar_dash.svg")
