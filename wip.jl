cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,StatsPlots,Distributions,Hungarian,LsqFit,LinearAlgebra,LambertW
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

## Computing <v²(t)>
# load("data/locations_single_default_Var0.0_r10.jld","R")
# XY
L         = 200
    Q     = 1
    Rtilde= 40
    Reach = load("data/locations_single_default_Var0.0_r1.jld","R")
    save_every = load("data/locations_single_default_Var0.0_r1.jld","save_every")
    dt = determine_dt(0,0)
    tmax = load("data/locations_single_default_Var0.0_r1.jld","tmax")
    Ts,Vars = load("data/locations_single_default_Var0.0_r1.jld","Ts","Vars")

MSV_avgs = Array{Number,4}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),Rtilde))
MSV_vars = Array{Number,4}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),Rtilde))
z= @elapsed for r in 1:Rtilde
    println("r = $r")
    locations = load("data/locations_single_default_Var0.0_r$r.jld","locations")
    loc = convert(Array{Union{Tuple{Int16,Int16},Missing}},locations)
    MSV_avgs[:,:,:,r],MSV_vars[:,:,:,r] = SV(L,length(Ts),length(Vars),Reach,loc,save_every*dt)
end
MSV_avg = mean(MSV_avgs,dims=4)[:,:,:,1]
MSV_avg2 = mean(MSV_avg[2:end,:,:],dims=1)[1,:,:]
plot(Ts,MSV_avg2[:,1],m=:.,xlabel=L"T",ylabel=L"⟨v²⟩")
plot!(Ts,0.6Ts,c=:black)
L         = 200
    Q     = 1
    Rtilde= 36
    T = 0
    Reach = load("data/locations_single_default_T$(T)_r1.jld","R")
    save_every = load("data/locations_single_default_T$(T)_r1.jld","save_every")
    dt = determine_dt(0,0)
    tmax = load("data/locations_single_default_T$(T)_r1.jld","tmax")
    Ts,Vars = load("data/locations_single_default_T$(T)_r1.jld","Ts","Vars")

MSV_avgs = Array{Number,4}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),Rtilde))
MSV_vars = Array{Number,4}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),Rtilde))
z= @elapsed for r in 1:Rtilde
    println("r = $r")
    locations = load("data/locations_single_default_T$(T)_r$r.jld","locations")
    loc = convert(Array{Union{Tuple{Int16,Int16},Missing}},locations)
    MSV_avgs[:,:,:,r],MSV_vars[:,:,:,r] = SV(L,length(Ts),length(Vars),Reach,loc,save_every*dt)
end
MSV_avg = mean(MSV_avgs,dims=4)[:,:,:,1]
MSV_std = sqrt.(mean(MSV_vars,dims=4)[:,:,:,1]/Rtilde)

MSV_avg2 = mean(MSV_avg[2:end,:,:],dims=1)[1,:,:]

p=plot(Vars,MSV_avg2[1,:],m=:.,xlabel=L"σ²",ylabel=L"⟨v_{\!∞}^2⟩",label="T=$T")
ylims!(-0.01,0.2)
ylabel!(L"⟨⟨v²(t)⟩_{real}⟩_{time}")

plot(Array(0:save_every:tmax/dt)*dt,MSV_avg[:,1,1])
