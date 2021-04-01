## XY annihilation V2.0
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra,LambertW
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

L = 100
r0 = 36
R = 120
T = 0.1
Var = 0.0

tmax = 3000 ; dt = determine_dt(T,Var) ; save_every = 1
tt = Array(0:dt:tmax)

history_distance = Matrix{Union{Float64,Missing}}(undef,(1+floor(Int16,floor(Int,tmax/dt)/save_every),R))
history_location = Array{Tuple{Union{Int16,Missing},Union{Int16,Missing}},3}(undef,(2,1+floor(Int16,floor(Int,tmax/dt)/save_every),R))

c = 0
z = @elapsed Threads.@threads for r in 1:R
    global c += 1 ; println("Realisation $c/$R started.")
    thetas = create_pair_vortices_v2(L,r0)
    omegas = rand(Normal(0,sqrt(Var)),L,L)
    history_location[:,:,r],history_distance[:,r] = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)
end
println("Runtime = $(round(Int,z)) seconds")

# save("data\\XY_annihilation.jld","L",L,"r0",r0,"R",R,"T",T,"Var",Var,"tmax",tmax,"dt",dt,"save_every",save_every,"comment","history_distance[ii:end] .= 0.0 enabled","history_distance",history_distance)
L,r0,R,T,Var,tmax,dt,save_every,history_distance = load("data\\XY_annihilation.jld","L","r0","R","T","Var","tmax","dt","save_every","history_distance")
tt = Array(0:dt:tmax)

history_distance_avg = -ones(size(history_distance)[1])
for i in 1:size(history_distance)[1]
    history_distance_avg[i] = mean(skipmissing(history_distance[i,:]))
end
#
# # sum(history_distance .=== missing)
# plot(tt,history_distance_avg)


r0s = [20,30]
    tannihilations = [310,750]
    # tannihilations = [375,841] # from mean annihilation times (doesn't work)
    ind = zeros(Int,2)
    for i in eachindex(r0s)
        ind[i] = findfirst(x->x<r0s[i],history_distance_avg)
    end
    # lastind = round.(Int,[350,100,100]/dt)
    rt1 = plot(xlabel=L"t")
    for i in eachindex(r0s)
        indice = ind[i] ; r0 = r0s[i]
        plot!(tt[indice:end] .- tt[indice],history_distance_avg[indice:end],c=i)

        times = Array(0:dt:tannihilations[i])
        a =  r0^2*(2log(r0)-1)/4/tannihilations[i]
        plot!(times,exp.(0.5(1 .+ lambertw.(4exp(-1)*a*(tannihilations[i] .- times)))),line=:dash,c=:black)
    end
    xlims!(0,1500)


L,R,r0s,TV,tmax2,save_every2,locations,distance = JLD.load("../vortices_study/data_vortices/pair_defaults_PBC_L200_T0.1_Var0.1.jld","L","R","r0s","TV","tmax","save_every","locations","distance")
    nR0 = 3
        distance_avg  = Array{Float64}(undef,(length(0:save_every2:tmax2/dt),length(TV),nR0)) ; distance_std  = Array{Float64}(undef,(length(0:save_every2:tmax2/dt),length(TV),nR0))
        for n in 1:size(distance)[3]
            for i in 1:size(distance)[2]
                for t in 1:size(distance)[1]
                    distance_avg[t,i,n] = mean(skipmissing(distance[t,i,n,:]))
                    distance_std[t,i,n] = std(skipmissing(distance[t,i,n,:]))
                end
            end
        end
        every_plot = 5
        tt2 = Array(0:dt*every_plot*save_every2:tmax2)
        # plot!(tt2,distance_avg[1:2:end,1,1],line=:dot,c=1)
        plot!(tt2,distance_avg[1:every_plot:end,1,2],line=:dash,c=1)
        plot!(tt2,distance_avg[1:every_plot:end,1,3],line=:dash,c=2)
        xlims!((0,1000))
        ylims!((-1,38))
        annotate!(20,35.5,text(L"⟨R(t)⟩",15,:left,:black))
        # savefig("../poster/rt.pdf")
        # annotate!(850,38.5*92/100,text("(a)",15,:left,:black))
&
# # To show that we do recover the results à la Yurke
#
# r0s = [10,20,30]
#     t0s = zeros(Int,3,R) # departure from r0 (indices)
#     t1s = zeros(Int,R) # annihilation time (indices)
#     for i in eachindex(r0s)
#         for r in 1:R
#             t0s[i,r] = findfirst(x->x<r0s[i],history_distance[:,r])
#             t1s[r]   = findfirst(x->x==0,history_distance[:,r])
#         end
#     end
# history_distance_rvs = Array{Union{Missing, Float64},2}(missing,size(history_distance))
# for r in 1:R
#     history_distance_rvs[1:t1s[r],r] = reverse(history_distance[1:t1s[r],r])
# end
# history_distance_rvs_avg = zeros(size(history_distance_rvs)[1])
# for i in 1:size(history_distance_rvs)[1]
#     history_distance_rvs_avg[i] = mean(skipmissing(history_distance_rvs[i,:]))
# end
# plot(xlabel=L"\tilde{t}",ylabel=L"R(\tilde{t})")
#     plot!(tt[2:end],history_distance_rvs_avg[2:end],axis=:log,lw=2)
#     xlims!(0.1,1000)
#     plot!(tt[2:end],1exp.(lambertw.(2pi*tt[2:end])/2),c=:black,line=:dash)
#     plot!(tt[2:end],sqrt.(tt[2:end]),c=:green,line=:red)
#     annotate!(10,1.5,text(L"\sim \tilde{t}\,^{1/2}",12))
#     yticks!([1,2,3,4,5,6,7,8,9,10,20,30,40],string.([1,2,3,4,"",6,"",8,"",10,20,30,40]))
# savefig("figures_draft\\XY_annihilation_yurke.pdf")
#
#
# L,TV,r0s,tmax,R,rt = load("data_vortices/rt_r0_tmax_tmax500.jld","L","TV","r0s","tmax","R","rt")
# rt_avg = Matrix{Number}(undef,size(rt)[1:2]) ; rt_std = Matrix{Number}(undef,size(rt)[1:2])
# for i in eachindex(TV)
#     for k in i:length(TV):length(r0s)
#         rt_avg[i,ceil(Int,k/length(TV))] = mean(skipmissing(rt[i,ceil(Int,k/length(TV)),:]))
#         rt_std[i,ceil(Int,k/length(TV))] = std(skipmissing(rt[i,ceil(Int,k/length(TV)),:]))
#     end
# end
#
# TV_string = [("0.10","0.10"),("0.05","0.05"),("0.05","0.10"),("0.10","0.05")]
# rt2 = plot(xlabel=L"R_0",xlims=(0,L/2),ylims=(0,L/2),size=(400,400),legend=:bottomright)
#     markers = [:circle,:square,:utriangle,:dtriangle]
#     scatter!((NaN,NaN),label="T , σ² = ",m=nothing)
#     for i in 1:length(TV) scatter!(r0s[i:length(TV):length(r0s)],rt_avg[i,:],m=markers[i],ms=6,c=i,label="$(TV_string[i][1]) , $(TV_string[i][2])") end
#     plot!([0,L/2],[0,L/2],c=:black)
#     annotate!(3,93,text(L"⟨R_{t=500}⟩",:black,15,:left))
#     annotate!(70,92,text("(b)",15,:left,:black))
#
# rt = plot(rt1,rt2,layout=2,size=(800,400))
# # savefig(rt,"figures_draft/rt_r0.pdf")
