## No difference +/- 1 defects
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,Distributions,Hungarian,LsqFit,LinearAlgebra
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

L,R,r0s,TV,tmax2,save_every2,locations,distance = JLD.load("../vortices_study/data_vortices/pair_defaults_PBC_L200_T0.1_Var0.1.jld","L","R","r0s","TV","tmax","save_every","locations","distance")
gg = zeros(Union{Missing, Float64},(2,1274,100))
# NB : locations = 2×1274×1×3×100 Array{Tuple{Union{Missing, Int16},Union{Missing, Int16}},5}:
for i in 1:100
    for j in 1:1274
        gg[1,j,i] = dist_missing(locations[1,j,1,1,i] , locations[1,1,1,1,i],200).^2
        gg[2,j,i] = dist_missing(locations[2,j,1,1,i] , locations[2,1,1,1,i],200).^2
    end
end
gg_avg = zeros(2,1274)
for j in 1:1274
    gg_avg[1,j] = mean(skipmissing(gg[1,j,:]))
    gg_avg[2,j] = mean(skipmissing(gg[2,j,:]))
end
plot(gg_avg[1,2:end],axis=:log)
plot!(gg_avg[2,2:end])
plot!(0.05Array(1:1200).^1.5)

## Films pour SM
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,Distributions,Hungarian,LsqFit,LinearAlgebra
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)
cols = [:white,:blue,:green,:orange,:red,:white]

L=200
r0 = 50
T = 0.1
Var = 0.

    # transients
thetas = create_pair_vortices_v2(L,r0)
omegas = rand(Normal(0,sqrt(Var)),L,L)
t = 0.0
while t<200
    global t += dt
    global thetas = update(thetas,omegas,L,T,dt)
end

    # record
tmax = 4000 ; dt = determine_dt(T,Var) ; save_every = 50 ; imax = round(Int,tmax/dt/save_every)
thetas_hist = Vector{Array{Float64,2}}(undef,imax)
z1 = @elapsed for i in 1:imax
    for j in 1:save_every
        global thetas = update(thetas,omegas,L,T,dt)
    end
    thetas_hist[i] = thetas
end
println("Runtime = $(round(Int,z1)) seconds")


traj = Matrix{Tuple{Int64,Int64}}(undef,(2,imax))
imaxi = imax
for i in 1:imax
    a,b = spot_vortex_antivortex(thetas_hist[i])
    try
        traj[1,i] = a[1]
        traj[2,i] = b[1]
    catch
        global imaxi = i-1 # annihilation frame
        break
    end
end
println("$imaxi/$imax")
z = @elapsed anim = @animate for i in 1:imaxi
    thetas = thetas_hist[i]
    heatmap(mod.(thetas',2π),c=cgrad(cols),colorbar=nothing,size=(512,512))

    plot!(traj[1,1:imaxi],c=:black)
    plot!(traj[2,1:imaxi],c=:black,line=:dot)
    for vortex in spot_vortices(thetas_hist[i])
        if vortex[3]>0 scatter!((vortex[1:2]).+(0.5,0.5),m=:circle,c=:black,ms=9)
        else scatter!((vortex[1:2]).+(0.5,0.5),m = (8, 9.0, :circle,:transparent, stroke(3, :black)))
        end
    end
    xlims!((1,200))
    ylims!((1,200))
    annotate!(165,185,text(L"t="*string(round(Int,i*dt*save_every)),15,:black))
end
println("Runtime = $(round(Int,z)) seconds")
# mp4(anim,"figures_draft\\film_SM_ActiveXY.mp4",fps=25)
mp4(anim,"figures_draft\\film_SM_XY_annihilation_.mp4",fps=25)



##
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra,LambertW
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

data = load("data/Horizontal_Vortices_L200_LowTemp.jld")
    L    = data["L"]
    R    = data["R"]
    Ts   = data["Ts"]
    Vars = data["Vars"]
    number_free_vortex = data["number_free_vortex"] ; number_free_vortex_avg = mean(number_free_vortex,dims=4) ; number_free_vortex_std = std(number_free_vortex,dims=4)

p2b = plot(xlabel="T",legend=:left,size=(400,400))
    for i in eachindex(Vars)
        plot!(Ts,(number_free_vortex_avg[1:end,i,end,1]),m=:.,ms=3,rib=0number_free_vortex_std[:,i,end,1],line=nothing,c=i)
    end
    p0 = [11,0.7,0.5] # beta, Tc, nu
    @. model3(x, p) = (x .> p[2]).*exp(p[1]*((Complex(x).-p[2])/p[2]).^p[3])
    @. model2(x, p) = (x .> p[2]).*exp(p[1]*((Complex(x).-p[2])/p[2]).^0.5)

    # Fit blue
    col = 1
    plot!([NaN,NaN],[NaN,NaN],rib=0,c=col,label="σ² = $(Vars[col])")

    firstind = 5
    fit2 = curve_fit(model2, Ts[1:end-firstind], number_free_vortex_avg[1:end-firstind,col,end,1], p0[1:2])
    fit3 = curve_fit(model3, Ts[1:end-firstind], number_free_vortex_avg[1:end-firstind,col,end,1], p0)
    p3 = string.(round.(coef(fit3),digits=2))
    # plot!(Ts,real.(model3(Ts,coef(fit3))),c=col)
    plot!(Ts,real.(model2(Ts,coef(fit2))),c=col,line=:solid)
    # plot!(Ts,real.(model(Ts,coef(fit))),c=col,label=L"\beta = "*p[1]*L", T_c = "*p[2]*L", \nu = "*p[3])
    # scatter!((Ts[end-firstind],number_free_vortex_avg[end-firstind,col,end,1]),c=col,ms=6)

    # Fit orange
    col = 2
    firstind = 12
    fit2 = curve_fit(model2, Ts[1:end-firstind], number_free_vortex_avg[1:end-firstind,col,end,1], p0[1:2])
    fit3 = curve_fit(model3, Ts[1:end-firstind], number_free_vortex_avg[1:end-firstind,col,end,1], p0)
    p = string.(round.(coef(fit3),digits=2))
    plot!([NaN,NaN],[NaN,NaN],rib=0,c=col,label="σ² = $(Vars[col])")
    # plot!(Ts,real.(model3(Ts,coef(fit3))),c=col)
    plot!(Ts,real.(model2(Ts,coef(fit2))),c=col,line=:solid)
    # plot!(Ts,real.(model3(Ts,coef(fit))),c=col,label=L"\beta = "*p[1]*L", T_c = "*p[2]*L", \nu = "*p[3])
    # scatter!((Ts[end-firstind],number_free_vortex_avg[end-firstind,col,end,1]),c=col,ms=6)

    # Fit green
    col = 3
    firstind = 18
    fit2 = curve_fit(model2, Ts[1:end-firstind], number_free_vortex_avg[1:end-firstind,col,end,1], p0[1:2])
    fit3 = curve_fit(model3, Ts[1:end-firstind], number_free_vortex_avg[1:end-firstind,col,end,1], p0)
    p = string.(round.(coef(fit3),digits=2))
    plot!([NaN,NaN],[NaN,NaN],rib=0,c=col,label="σ² = $(Vars[col])")
    # plot!(Ts,real.(model3(Ts,coef(fit3))),c=col)
    plot!(Ts,real.(model2(Ts,coef(fit2))),c=col,line=:solid)
    # println("Fits")
    # println("2 : ",round.(coef(fit2),digits=2)," gof ", sum(abs.(fit2.resid[1:end-firstind]))/2/length(fit2.resid[1:end-firstind]))
    # println("3 : ",round.(coef(fit3),digits=2)," gof ", sum(abs.(fit3.resid[1:end-firstind]))/3/length(fit3.resid[1:end-firstind]))


    # plot!(Ts,real.(model3(Ts,coef(fit))),c=col,label=L"\beta = "*p[1]*L", T_c = "*p[2]*L", \nu = "*p[3])
    # scatter!((Ts[end-firstind],number_free_vortex_avg[end-firstind,col,end,1]),c=col,ms=6)
    xlims!((-0.050,1.05))
    ylims!((-10,250))
    annotate!(0.05,230,text(L"n_v",15))
    annotate!(.97,10,text("(b)",12))
    # plot!([NaN,NaN],c=:grey,label="Fits "*L"\exp\left(α \sqrt{\frac{T-T_c}{T_c}}\,\right)")
    # plot!([NaN,NaN],line=:solid,c=:grey,label="Fit 3-param")


c=5.16
    L = load("data/Vortices_PhaseSpace_L200_LowTemp.jld","L")
    Vars = load("data/Vortices_PhaseSpace_L200_LowTemp.jld","Vars")
    Ts = load("data/Vortices_PhaseSpace_L200_LowTemp.jld","Ts")
    init = load("data/Vortices_PhaseSpace_L200_LowTemp.jld","init")
    number_free_vortex_avg = mean(load("data/Vortices_PhaseSpace_L200_LowTemp.jld","number_free_vortex"),dims=4)
    # number_all_vortex_avg = mean(load("data/Vortices_PhaseSpace_L200_LowTemp.jld","number_all_vortex"),dims=4)

    p2a=plot(xlabel="σ²",ylabel="T",size=(600,400),yguidefontrotation=-90)
    heatmap!((Vars),Ts,log10.(number_free_vortex_avg[:,:,end,1] .+1),c=cgrad([:blue,:green,:orange,:red]),interpolate = :true,colorbar_title=L"\log_{10}(n_v+1)")
    Tkt = 0.82 ; xx= Array(0:0.0001:7 /log(L)^2) ; plot!(xx,Tkt*(1 .- sqrt.(xx)*log(L)/sqrt(c)),c=:white,lw=2)
    xlims!((0,.25))
    ylims!((0,1))
    annotate!(0.225,0.075,text("(a)", 14, :black))
    # annotate!(0.135,0.85,text(L"T\!_c\,(\sigma) = T\!_{KT}\,(1-a\,\sigma\,\ln\, L)",15,:white))

plot(p2a,p2b,layout=2,size=(900,400))
# savefig("figures_draft\\free_vortices.svg")
# savefig("figures_draft\\free_vortices.pdf")

## New method for generating vortices configurations
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra,LambertW
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

L = 32
r0 = 16
grille = [(i,j) for i=1:L, j=1:L]
xs     = [grille[n][1] for n in 1:length(grille)]
ys     = [grille[n][2] for n in 1:length(grille)]

thetas = create_pair_vortices_v2(L,r0)

pp = quiver(xs,ys,quiver=(vec(cos.(thetas)),vec(sin.(thetas))),size=(500,500),color=:black)
    scatter!((16,24),m=:circle,c=:blue,ms=12)
    scatter!((16,8),m = (8, 12.0, :circle,:white, stroke(3, :blue)))
    xticks!([0,10,20,30],string.([0,10,20,30]))
# savefig("figures_draft/manual_double_vortex_relaxed.pdf")
# savefig("figures_draft/manual_double_vortex_not_relaxed.pdf")

thetas = create_isolated_vortex(L,L,1)
pp = quiver(xs,ys,quiver=(vec(cos.(thetas)),vec(sin.(thetas))),size=(500,500),color=:black)
    scatter!((16,16),m=:circle,c=:blue,ms=12)
    xticks!([0,10,20,30],string.([0,10,20,30]))
    # savefig("figures_draft/manual_single_vortex.pdf")

## Film for vortex displacement on boundary
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra,LambertW
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
gr(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\traj_T0.05_Var0.1_champ.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
# smoothedE = (energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
debut = 4000
fin   = 9999
traj = Vector{Tuple{Int64,Int64}}(undef,fin-debut+1)
for i in debut:fin
    a,b = spot_vortex_antivortex(thetas_hist[floor(Int,i/every)])
    traj[i-debut+1] = b[1]
end

cols = [:white,:blue,:green,:orange,:red,:white]

z = @elapsed anim = @animate for i in debut:20:fin
    thetas = thetas_hist[floor(Int,i/every)]
    heatmap(mod.(thetas',2π),c=cgrad(cols),colorbar=nothing,size=(1024,1024))

    plot!(traj[1:i-debut+1],c=:black,line=:dot)
    plot!(traj[i-debut+1:end],c=:black)
    for vortex in spot_vortices(thetas_hist[floor(Int,i/every)])
        if vortex[3]<0
            scatter!((vortex[1:2]).+(0.5,0.5),m=:circle,c=:black,ms=9)
        end
    end
    xlims!((1,200))
    ylims!((1,200))
end
mp4(anim,"../presentation\\figures\\test_film_fast.mp4",fps=60)

## C(t,tw)
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

L,Ts,Vars,tmax,Ctw,tws,tsave = load("data/Ctw_L200_HighTemp.jld","L","Ts","Vars","tmax","Ctw","tws","tsave")
Ctw_avg = mean(Ctw,dims=5)
Vars
Ts
tsave
tws
pa = plot(xaxis=:log,xlabel=L"t-t_w",ylabel=L"C(t,t_w)")
    for i in 1:6
        tw  = tws[i]
        ind = findfirst(x->x>tw,tsave)
        plot!(tsave[ind:end].-tw,Ctw_avg[3,1,i,ind:end,1])
    end
    annotate!(1.85,0.14,text(L"T\,=0.4",12,:black,:left))
    annotate!(1.5,0.06,text(L"σ²= 0.0",12,:black,:left))
    annotate!(10000,0.77,text("(a)",15))

pb = plot(xaxis=:log,xlabel=L"t-t_w",ylabel=L"C(t,t_w)")
    for i in 1:6
        tw  = tws[i]
        ind = findfirst(x->x>tw,tsave)
        plot!(tsave[ind:end].-tw,Ctw_avg[1,3,i,ind:end,1])
    end
    annotate!(1.85,0,text(L"T\,=0.0",12,:black,:left))
    annotate!(1.5,-0.11,text(L"σ²= 0.1",12,:black,:left))
    annotate!(10000,0.9,text("(b)",15))

pc = plot(xaxis=:log,xlabel=L"t-t_w",ylabel=L"C(t,t_w)")
    for i in 1:6
        tw  = tws[i]
        ind = findfirst(x->x>tw,tsave)
        plot!(tsave[ind:end].-tw,Ctw_avg[2,2,i,ind:end,1])
    end
    annotate!(1.85,0.02,text(L"T\,=0.1",12,:black,:left))
    annotate!(1.5,-0.09,text(L"σ²= 0.05",12,:black,:left))
    annotate!(10000,0.87,text("(c)",15))


pd = plot(xaxis=:log,xlabel=L"t-t_w",ylabel=L"C(t,t_w)")
    for i in 1:6
        tw  = tws[i]
        ind = findfirst(x->x>tw,tsave)
        plot!(tsave[ind:end].-tw,Ctw_avg[3,3,i,ind:end,1])
    end
    annotate!(1.85,0.06,text(L"T\,=0.4",12,:black,:left))
    annotate!(1.5,-0.03,text(L"σ²= 0.1",12,:black,:left))
    annotate!(10000,0.75,text("(d)",15))


p = plot(pa,pb,pc,pd,layout=4,size=(800,800))
# savefig("figures_draft\\Ctw.pdf")


## Energy dissipation
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,size=(400,400),thickness_scaling = 1.5)

T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\traj_T0.05_Var0.1_champ.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
# smoothedE = (energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
debut = 4000
mid   = 7000
fin   = 9000
energies = Vector{Float64}(undef,fin-debut+1)
traj = Vector{Tuple{Int64,Int64}}(undef,fin-debut+1)
for i in debut:fin
    a,b = spot_vortex_antivortex(thetas_hist[floor(Int,i/every)])
    traj[i-debut+1] = b[1]
end
spins_to_consider = zeros(Int,L,L)
for element in traj
    x,y = element
    tkns = 6
    for i in x-tkns:x+tkns
        for j in y-tkns:y+tkns
            spins_to_consider[i,j] = 1
        end
    end
end
# for i in 1:L
#     for j in 1:L
#         a = (traj[1][2]-traj[end][2])/(traj[1][1]-traj[end][1])
#         b = traj[end][2]-traj[end][1]a
#         if traj[end][1]<i<traj[1][1] && a*i + b - 10 <j< a*i + b + 10 spins_to_consider[i,j] = 1 end
#     end
# end

sum(spins_to_consider)
delim = Tuple{Int64,Int64}[]
for i in 1:L
    for j in 1:L
        if spins_to_consider[i,j] == 1
            push!(delim,(i,j))
        end
    end
end

scatter!(delim,c=:black,ms=.5)
savefig("figures/zone_dE.pdf")

@time for i in debut:fin
    thetas = thetas_hist[floor(Int,i/every)]
    tmp = energy(thetas)[1]
    energies[i-debut+1] = sum(spins_to_consider.*tmp)
end

ee = energies[1:fin-debut]/sum(spins_to_consider)
    plot(xlabel="t",ylabel="E/L²",size=(400,400))
    plot!(Array(debut+1:fin)*0.157,ee)
    plot!(fill(mid*0.157,2),[-3.75,-3.815],line=:dash)
    plot!(fill(fin*0.157,2),[-3.75,-3.815],line=:dash)
    # savefig("figures\\dissipation_along_traj2.pdf")

## ⟨r(t)⟩ vs r0
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

L = 200
    TV = [(0.1,0.1),(0.05,0.05),(0.05,0.1),(0.1,0.05)]
    r0s = Array(8:4:Int(L/2)) ; @assert mod(length(r0s),length(TV)) == 0
    tmax = 500 ; save_every = 500
    dt = determine_dt(TV)
    R = 48
    rt = Array{Any,3}(missing,length(TV),Int(length(r0s)/length(TV)),R)

z = @elapsed for i in eachindex(TV)
    for k in i:length(TV):length(r0s)
        Threads.@threads for r in 1:R
            T,Var = TV[i]; r0 = r0s[k]
            println("r0 = $r0 , T = $(Float16(T)) , σ² = $(Float16(Var)) , r = $r")
            thetas = create_pair_vortices(L,r0+2)
            omegas = rand(Normal(0,sqrt(Var)),L,L)

            rt[i,ceil(Int,k/length(TV)),r] = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)[2][end] # either number or missing
        end
    end
end
println("Runtime = $(round(Int,z)) seconds")
# save("data_vortices/rt_r0_tmax_tmax500.jld","L",L,"TV",TV,"r0s",r0s,"tmax",tmax,"R",R,"rt",rt)

# L,TV,r0s,tmax,R,rt = load("data_vortices/rt_r0_tmax500.jld","L","TV","r0s","tmax","R","rt")
rt_avg = Matrix{Number}(undef,size(rt)[1:2]) ; rt_std = Matrix{Number}(undef,size(rt)[1:2])
for i in eachindex(TV)
    for k in i:length(TV):length(r0s)
        rt_avg[i,ceil(Int,k/length(TV))] = mean(skipmissing(rt[i,ceil(Int,k/length(TV)),:]))
        rt_std[i,ceil(Int,k/length(TV))] = std(skipmissing(rt[i,ceil(Int,k/length(TV)),:]))
    end
end

TV_string = [("0.10","0.10"),("0.05","0.05"),("0.05","0.10"),("0.10","0.05")]
plot(xlabel=L"r_0",xlims=(0,L/2),ylims=(0,L/2),size=(400,400),legend=:bottomright)
    markers = [:circle,:square,:utriangle,:dtriangle]
    scatter!((NaN,NaN),label="T , σ² = ",m=nothing)
    for i in 1:length(TV) scatter!(r0s[i:length(TV):length(r0s)],rt_avg[i,:],m=markers[i],ms=6,c=i,label="$(TV_string[i][1]) , $(TV_string[i][2])") end
    plot!([0,L/2],[0,L/2],c=:black)
    annotate!(3,93,text(L"⟨r(t)⟩",:black,15,:left))
    # savefig("figures_draft\\rt_r0.pdf")


## Calcul de l'energie d'un single vortex
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

Ls = Array(25:25:200)
    R = 90
    T = 0.05
    Var_base = 0.1 ; L_base = 200
    dt = determine_dt(0,0)
    ttr = 500 ; save_every = 25 ; tmax = 600
    tsave = Array(ttr + save_every : save_every : tmax)

dE = fill(NaN,length(Ls),length(tsave),R)

z = @elapsed for i in eachindex(Ls)
    L = Ls[i]
    Var = Var_base * (log(L_base)/log(L))^2
    println("L=$L ,  Var = $Var")
    Threads.@threads for r in 1:R
        thetas_vortex  = create_isolated_vortex(L,L,+1)
        thetas_control = create_Lattice(L,T,Var,"LowTemp")[1][:,:,1]
        omegas = rand(Normal(0,sqrt(Var)),L,L)
        t = 0.0 ; token_time = 1
        while t<ttr
            thetas_vortex  = update(thetas_vortex,omegas,L,T,dt)
            thetas_control = update(thetas_control,omegas,L,T,dt)
            t += dt
        end
        while t<tmax
            thetas_vortex  = update(thetas_vortex,omegas,L,T,dt)
            thetas_control = update(thetas_control,omegas,L,T,dt)
            t += dt
            if round(Int,t) == tsave[token_time]
                dE[i,token_time,r] = energy(thetas_vortex)[2] - energy(thetas_control)[2]
                token_time = min(token_time + 1,length(tsave))
            end
        end

    end
end
println("Runtime = $(round(Int,z)) seconds")


# save("data_vortices\\energie_Ls.jld","Ls",Ls,"dE",dE,"R",R,"T",T,"Var_base",Var_base,"L_base",L_base,"tmax",tmax,"ttr",ttr,"save_every",save_every)
Ls,dE,R,T,Var_base,L_base,tmax,ttr,save_every = load("data_vortices\\energie_Ls.jld","Ls","dE","R","T","Var_base","L_base","tmax","ttr","save_every")

dE_avg = mean(dE,dims=3) ; dE_std = std(dE,dims=3)
p=plot(uaxis=:log,xlabel=L"L",ylabel=L"dE")
    for i in 1:length(tsave)
        plot!(Ls[2:end],dE_avg[2:end,i,1],rib=0dE_std[:,i,1],label="t = $(tsave[i])",m=:circle)
    end
    plot!(Ls[2:end],1.38sqrt.(Ls[2:end]) .* log.(Ls[2:end]),c=:black,line=:solid,label=L"\ln(L) \sqrt{L}")
    savefig("figures\\dE_Ls.pdf")

p=plot(uaxis=:log,xlabel=L"L",ylabel=L"dE")
    plot!(Ls[2:end],vec(mean(dE_avg[2:end,:,1],dims=2)),m=:circle,label="σ² = 0.025")
    plot!(Ls[2:end],1.38sqrt.(Ls[2:end]) .* log.(Ls[2:end]),c=:black,line=:solid,label=L"\ln(L) \sqrt{L}")
    savefig("figures\\dE_Ls.pdf")

## D(T) Dependence (Single Vortex)
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian,LsqFit,LinearAlgebra
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)
include("IDrealisation.jl")

L         = 200
    Q     = 1
    R     = 12
    Ts    = Array(0:0.01:0.4)
    Vars  = [0.025]
    tmax  = Int(2000) ; save_every = 10
    dt    = determine_dt(0,0)

locations_single_vortex = Array{Union{Tuple{Int,Int},Missing}}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),R))
z = @elapsed Threads.@threads for rr in 1:R
    for i in eachindex(Ts)
        for j in eachindex(Vars)
            T = Ts[i] ; Var = Vars[j]
            println("r = $r , T = $(Float16(T)) , σ² = $(Float16(Var))")
            thetas = create_isolated_vortex(L,L,Q)
            omegas = rand(Normal(0,sqrt(Var)),L,L)

            t = 0.0
            locations_single_vortex[:,i,j,rr] = evolve_FBC(Q,thetas,omegas,T,dt,tmax,save_every)
        end
    end
end
println("Runtime = $(round(Int,z)) seconds")
# JLD.save("data/locations_single_default_r$r.jld","L",L,"R",R,"Ts",Ts,"Vars",Vars,"tmax",tmax,"save_every",save_every,"locations",locations_single_default)

# JLD.save("../vortices_study/data_vortices/locations_single_default.jld","L",L,"R",R,"Ts",Ts,"Vars",Vars,"tmax",tmax,"save_every",save_every,"locations",locations_single_default)


L         = 200
    Q     = 1
    Rtilde= 33
    Reach = 12
    R = Reach*Rtilde
    Ts    = [0.05]
    Vars  = Array(0.025:0.025:0.2)
    # Ts    = Array(0:0.025:0.7)
    # Vars  = [0.0]
    # Ts    = Array(0:0.01:0.4)
    # Vars  = [0.025]
    tmax  = Int(2000) ; save_every = 10
    dt    = determine_dt(0,0)

#
MSD_avgs = Array{Number,4}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),Rtilde))
MSD_vars = Array{Number,4}(undef,(length(0:save_every:tmax/dt),length(Ts),length(Vars),Rtilde))
z= @elapsed for r in 1:Rtilde
    println("r = $r")
    locations = load("data/locations_single_default_T0.05_r$r.jld","locations")
    loc = convert(Array{Union{Tuple{Int16,Int16},Missing}},locations)
    MSD_avgs[:,:,:,r],MSD_vars[:,:,:,r] = SD(L,length(Ts),length(Vars),Reach,loc)
end
MSD_avg = mean(MSD_avgs,dims=4)[:,:,:,1]
MSD_std = sqrt.(mean(MSD_vars,dims=4)[:,:,:,1]/Rtilde)
save("data_vortices\\MSD_single_default_L200_T0.2_R$R.jld","L",L,"R",Reach*Rtilde,"save_every",save_every,"tmax",tmax,"dt",dt,"MSD_avg",MSD_avg,"MSD_std",MSD_std,"Ts",Ts,"Vars",Vars)
L,R,save_every,tmax,dt,MSD_avg,MSD_std,Ts = load("data_vortices\\MSD_single_default_L200_Var0.025_R420.jld","L","R","save_every","tmax","dt","MSD_avg","MSD_std","Ts")
L,R,save_every,tmax,dt,MSD_avg,MSD_std,Ts = load("data_vortices\\MSD_single_default_L200_T0_R432.jld","L","R","save_every","tmax","dt","MSD_avg","MSD_std","Ts")
L,R,save_every,tmax,dt,MSD_avg,MSD_std,Ts = load("data_vortices\\MSD_single_default_L200_T0.2_R456.jld","L","R","save_every","tmax","dt","MSD_avg","MSD_std","Ts")
L,R,save_every,tmax,dt,MSD_avg,MSD_std,Ts = load("data_vortices\\MSD_single_default_L200_Var0.0_R480.jld","L","R","save_every","tmax","dt","MSD_avg","MSD_std","Ts")
Vars = Array(0:0.01:0.2)
Vars = [0.025]
D     = fill(NaN,(length(Ts),length(Vars)))
D_std = fill(NaN,(length(Ts),length(Vars)))
# loc = convert(Array{Union{Tuple{Int16,Int16},Missing}},locations_single_vortex)
# MSD_avg,MSD_std = SD(L,length(Ts),length(Vars),R,loc)
#
p=plot(xlabel=L"t",ylabel=L"⟨(r(t)-r_0)^2⟩",axis=:log,title="σ² = 0.025")
    couleur = 1
    styles = [:solid,:dashdot,:dash]
    t = Array(0:save_every:tmax/dt)*dt
    tlogt = t ./ log.(t)
    for j in 1:length(Vars)
        for i in 1:length(Ts)
            plot!(t[2:end],remove_negative(MSD_avg[2:end,i,j],1/R),c=couleur)
            # plot!(t[2:end]*(sqrt(Vars[j])),remove_negative(MSD_avg[2:end,i,j],1/R),c=couleur)
            # plot!(t[2:end]*Ts[i],remove_negative(MSD_avg[2:end,i,j],1/R),c=couleur)
            global couleur += 1
            # @. model(t, p) = 4*p[1]*t/log(t)
            @. model(t, p) = 4*p[1]*(sqrt(Vars[j])*t).^1.5
            p0 = [0.5]
            try
                fitt = curve_fit(model, t[10:500], remove_negative(MSD_avg[10:500,i,j],1/R), p0)
                println(coef(fitt))
                D[i,j] = coef(fitt)[1]
                # D_std[i,j] = maximum(abs.(confidence_interval(fitt, 0.05)[1] .- D[i,j]))
            catch e
                println(e)
                D[i,j] = NaN
                D_std[i,j] = NaN
            end
        end
    end
    plot!(t[20:end],.1t[20:end].^1.5,line=:dash,c=:black,label="Slope 3/2")
    # plot!(t[2:100],3t[2:100].^1,line=:solid,c=:black,label="Slope 1")
    display(p)
    # plot!(t[50:end],1.7t[50:end]./log.(t[50:end]),c=:black,label=L"∼ t^{1.5}")
savefig("figures\\MSD_NKM_Var0.025.pdf")

plot(xlabel="T",ylabel=L"D(T)_{NKM}")
    plot!(Ts[7:end],(D[7:end]),m=:circle,label="σ² = 0.025")
    # plot!(Ts[1:end],0.6Ts.^1.33,c=:black,line=:solid,label=L"0.6\,T^{4/3}")
    # plot!(Ts[1:end],0.45Ts,c=:black,line=:dash,label=L"0.45\,T")
    plot!(Ts[1:end],0.045 .+ 0.1Ts,c=:black,line=:dash,label="T/10 + 0.045")
    savefig("figures\\DT_NKM.pdf")

plot(xlabel="σ²",ylabel="D(T=0)")
    plot!(Vars[2:end],(D[2:end]),m=:circle)
savefig("figures\\DT_Kuramoto.pdf")


## ⟨r(t)⟩ = r0
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

# Parameters
L         = 200
    Q     = 1        # for a single vortex
    r0s   = Int.([10,20,30].+2) ; nR0 = length(r0s) # for two vortices
    R     = 100
    TV    = [(0.1,0.1)] ; T = Var = 0.1
    tmax  = Int(2000) ; save_every = 10
    dt    = determine_dt(0,0)

locations_pair_vortices = Array{Tuple{Union{Int16,Missing},Union{Int16,Missing}}}(undef,(2,length(0:save_every:tmax/dt),length(TV),nR0,R))
distance_pair_vortices  = Array{Union{Float64,Missing}}(undef,(length(0:save_every:tmax/dt),length(TV),nR0,R))
z = @elapsed for k in eachindex(r0s)
    for i in eachindex(TV)
        Threads.@threads for r in 1:R
            T,Var = TV[i]; r0 = r0s[k]
            println("r0 = $r0 , T = $(Float16(T)) , σ² = $(Float16(Var)) , r = $r")
            thetas = create_pair_vortices(L,r0)
            omegas = rand(Normal(0,sqrt(Var)),L,L)

            t = 0.0
            locations_pair_vortices[:,:,i,k,r],distance_pair_vortices[:,i,k,r] = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)
        end
    end
end
println("Runtime = $(round(Int,z)) seconds")
# JLD.save("../vortices_study/data_vortices/pair_defaults_PBC_test_.jld","L",L,"R",R,"r0s",r0s,"TV",TV,"tmax",tmax,"save_every",save_every,"locations",locations_pair_vortices,"distance",distance_pair_vortices)
# JLD.save("../vortices_study/data_vortices/pair_defaults_PBC_L$(L)_T$(T)_Var$Var.jld","L",L,"R",R,"r0s",r0s,"TV",TV,"tmax",tmax,"save_every",save_every,"locations",locations_pair_vortices,"distance",distance_pair_vortices)

L,R,r0s,TV,tmax,save_every,locations,distance = JLD.load("../vortices_study/data_vortices/pair_defaults_PBC_L200_T0.1_Var0.1.jld","L","R","r0s","TV","tmax","save_every","locations","distance")
nR0 = length(r0s)
# locations = locations_pair_vortices
# distance = distance_pair_vortices
realisations_controlled = Matrix{Any}(undef,length(TV),nR0) # controlled annihilation
realisations_uncontrolled = Matrix{Any}(undef,length(TV),nR0) # Unexpected and exit
distance_modified = copy(distance)
for tv in 1:length(TV)
    for nr0 in 1:nR0
        tmp_list_c = []
        tmp_list_u = []
        for r in 1:R
            # Test for controlled annihilation
                indc = findfirst(x->x==0,skipmissing(distance[:,tv,nr0,r]))
                if indc ≠ nothing push!(tmp_list_c,r) end
            # Test for exit situation
                indu = findfirst(x->ismissing(x),distance[:,tv,nr0,r])
                if indu ≠ nothing # either Unexpected or exit
                    last_loc1,last_loc2 = locations[:,indu-1,tv,nr0,r]
                    close_from_boundary1 = last_loc1[1] < 4 || last_loc1[1] > L - 4 || last_loc1[2] < 4 || last_loc1[2] > L - 4
                    close_from_boundary2 = last_loc2[1] < 4 || last_loc2[1] > L - 4 || last_loc2[2] < 4 || last_loc2[2] > L - 4
                    if close_from_boundary1 || close_from_boundary2 push!(tmp_list_u,r) end
                end
                # if indu ≠ nothing push!(tmp_list_c,r) end

        end
        realisations_controlled[tv,nr0] = tmp_list_c
        realisations_uncontrolled[tv,nr0] = tmp_list_u
        # @assert length(tmp_list_c) ≥ length(tmp_list_u)

        # Compensate exit by nothinging zeros
        token_max = length(tmp_list_c)
        println("ratio u/c = $(length(tmp_list_u)/length(tmp_list_c)) , token_max = $token_max")
        if token_max > 0
            token = 1
            for i in 1:length(realisations_uncontrolled[tv,nr0])
                real_pb = realisations_uncontrolled[tv,nr0][i]
                ind = findfirst(x->ismissing(x),distance[:,tv,nr0,real_pb])
                real_no_pb = realisations_controlled[tv,nr0][token]
                if token == token_max - 1 break ; else token += 1 end
                distance_modified[ind:end,tv,nr0,real_no_pb] .= missing
            end
        end
        # println(length(realisations_uncontrolled[tv,nr0]))
        # for i in 1:length(realisations_controlled[tv,nr0])
        #     rr = realisations_controlled[tv,nr0][i]
        #     ind = findfirst(x->x==0,distance[:,tv,nr0,rr])
        #     distance_modified[ind+1:end,tv,nr0,rr] .= missing
        # end
    end
end


distance_avg  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0)) ; distance_std  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0))
for n in 1:size(distance)[3]
    for i in 1:size(distance)[2]
        for t in 1:size(distance)[1]
            distance_avg[t,i,n] = mean(skipmissing(distance[t,i,n,:]))
            distance_std[t,i,n] = std(skipmissing(distance[t,i,n,:]))
        end
    end
end
distance_modified_avg  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0)) ; distance_modified_std  = Array{Float64}(undef,(length(0:save_every:tmax/dt),length(TV),nR0))
for n in 1:size(distance)[3]
    for i in 1:size(distance)[2]
        for t in 1:size(distance)[1]
            distance_modified_avg[t,i,n] = mean(skipmissing(distance_modified[t,i,n,:]))
            distance_modified_std[t,i,n] = std(skipmissing(distance_modified[t,i,n,:]))
        end
    end
end

tv = 1
    tt = Array(0:dt*save_every:tmax)
    p0 = [0.02]
    lastindex = [60,223,404]
    col = [:blue,:orangered1,:green]
    p  = plot(xlabel=L"t",ylabel=L"⟨r(t)⟩",legend=nothing)
    for n in 1:nR0
        r0 = r0s[n].-2
        # plot!(tt,distance_avg[:,1,n],rib=distance_std[:,1,n],label=L"r_0 = "*string(r0),c=n)
        # @. model(x, p) = r0*exp.(- (erfinv.(p[1]*x/r0)).^2)
        # ind = lastindex[n]
        # fitt = curve_fit(model, tt[1:ind], distance_avg[1:ind,1,n], p0)
        # plot!(tt[1:ind],model(tt[1:ind],coef(fitt)),c=:black)
        plot!(tt,distance_avg[:,tv,n],line=:solid,c=col[n],rib=0distance_std)
        # plot!(tt,0 .+distance_modified_avg[:,tv,n],line=:dash,c=col[n],rib=0distance_modified_std)
        end
        display(p)
        ylims!((-1,50))
        savefig("figures\\rt_PBC.pdf")
    # xlims!((-40,2000))


# Look at a few realisations
n = 25
    p=plot()
    for nn in 1:n
        plot!(tt,distance[:,1,3,nn])
    end
    plot!(tt,[mean(skipmissing(distance[i,1,3,1:n])) for i in 1:size(distance)[1]],c=:black)
    # xlims!(0,300)
    # ylims!(-10,50)
    display(p)

## Vizu accumulation angle differences
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

T = 0.0
Var = 0.1
L= 200
thetas = 2pi*rand(L,L)
omegas = rand(Normal(0,sqrt(Var)),L,L)
dt = determine_dt(T,Var)
t = 0.0

# tmax = 1000
# @time while t<tmax
#     global thetas = update(thetas,omegas,L,T,dt)
#     global t += dt
# end
# save("data\\vizu_accumulation.jld","thetas",thetas)
thetas = load("data\\vizu_accumulation.jld","thetas")
ℓ1 = 90
ℓ2 = 100
plot(yticks=nothing,xticks=1:10,grid=true,xlabel="Space [in units of lattice spacings]")
    i = 10
    scatter!((Array(1:ℓ2-ℓ1),fill(-0.2,ℓ2-ℓ1)),c=:black)
    quiver!(Array(1:ℓ2-ℓ1),fill(-0.2,ℓ2-ℓ1),quiver=(1 .*cos.(-thetas[i,1:ℓ]),0.2sin.(-thetas[i,1:ℓ])),c=:grey)
    for a in 1:10 annotate!(a,-0.42,text(string(round(mod(-thetas[i,a],2pi),digits=1)),:center,10,:blue)) end
    i = 90
    scatter!((Array(1:ℓ2-ℓ1),fill(0.25,ℓ2-ℓ1)),c=:black)
    quiver!(Array(1:ℓ2-ℓ1),fill(0.25,ℓ2-ℓ1),quiver=(1 .*cos.(thetas[i,1:ℓ]),0.2sin.(thetas[i,1:ℓ])),c=:grey)
    for a in 1:10 annotate!(a,0.12,text(string(round(mod(thetas[i,a],2pi),digits=1)),:center,10,:blue)) end

    plot!([-20,20],[0.05,0.05],c=:black)
    ylims!((-0.5,0.6))
    xlims!((-0.5,10.5))
    annotate!(0.,0.52,text("(a)",:center,15,:black))
    annotate!(0.,-0.05,text("(b)",:center,15,:black))
    annotate!(0.1,-0.42,text("θ =",:center,12,:blue))
    annotate!(0.1,0.12,text("θ =",:center,12,:blue))
    # savefig("figures_draft\\vizu_accumulation.pdf")


## Benchmark spot_pair_defaults_local_PBC
L     = 32
    r0    = 22
    T     = 0.1
    Var   = 0.1
    tmax  = Int(1E3) ; save_every = 1
    dt    = determine_dt(0,0)
    thetas = create_pair_vortices(L,r0)
    omegas = rand(Normal(0,sqrt(Var)),L,L)

     history_locations,history_distance = evolve_Pair(thetas,omegas,T,dt,tmax,save_every)
     history_distance[end]
     plot(history_locations[1,:])
     plot!(history_locations[2,:])
     xlims!((1,L))
     ylims!((1,L))


history_locations[:,end-1]
## Screenshot
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
# gr(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

T = 0.0
Var = 0.1
L= 200
thetas = 2pi*rand(L,L)
omegas = rand(Normal(0,sqrt(Var)),L,L)
dt = determine_dt(T,Var)
t = 0.0

tmax = 1000
@time while t<tmax
    global thetas = update(thetas,omegas,L,T,dt)
    global t += dt
end
ℓ = 10
quiver(1:ℓ,fill(1,ℓ),cos.(thetas[1,1:ℓ]),sin.(thetas[1,1:ℓ]))

# cols = [:white,:blue,:green,:orange,:red,:white]

p = plot(size=(400,400),axis=nothing,foreground_color_subplot=colorant"white")
    heatmap!(mod.(thetas',2π),c=cgrad(cols),colorbar=nothing,colorbar_title=L"\theta")
    for vortex in spot_vortices(thetas)
        if vortex[3] > 0 symb = :black else symb = :transparent end
        scatter!((vortex[1:2]).+(0.5,0.5), m = (8, 12.0, :circle,symb, stroke(2, :black)))
    end
    display(p)

    save("data\\NKM_realisation2.jld","Var",Var,"T",T,"L",200,"init","HighTemp","tmax",tmax,"thetas",thetas)


## Analytical argument
using QuadGK
using Plots,ColorSchemes,LaTeXStrings
pyplot(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

Vars = 0.005:0.01:1
esp = zeros(length(Vars))
for i in eachindex(Vars)
    σ = sqrt(Vars[i])
    # f(x) = asin(σ*x)*exp(-x^2)
    g(x) = cos(x)*x*exp(-(sin(x)/σ)^2)/σ/sqrt(pi)
    # esp[i] = quadgk(f,0,sin(1)/σ)[1]
    esp[i] = quadgk(g,0,1)[1]
end
plot(Vars,1 ./esp,axis=:log)
plot!(Vars,1 ./sqrt(Vars),axis=:log)

## Coarsening videos
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
gr(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)


L = 200
    T = 0.05
    Var = 0.2
    tmax = 1000
    init = "HighTemp"

    thetas = create_Lattice(L,T,Var,init)[1][:,:,1]
    omegas = rand(Normal(0,sqrt(Var)),L,L)
    dt = determine_dt(T,Var)

    n = floor(Int,tmax/dt)
    every = 10
    cols = [:white,:blue,:green,:orange,:red,:white]

    z = @elapsed anim = @animate for i in 1:every:n
        for e in 1:every global thetas = update(thetas,omegas,L,T,dt) end
        heatmap(mod.(thetas',2π),c=cgrad(cols),title="t = $(round(dt + dt*i,digits=1))")
    end
mp4(anim,"figures/gifs/coarsening_$(init)_T$(T)_Var$(Var).mp4",fps=10)

## Distribution G(x,t) de displacement des vortex at t0 t1 t2 t3 pour XY et NKM
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
# gr(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)
pyplot(box=true,size=400,400),fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)
include("IDrealisation.jl")


L = 100
    T = 0.2
    Var = 0.0
    r0 = (round(Int16,L/2),round(Int16,L/2)) # initial position
    tmax = 500
    t0 = 1
    nsave = 50
    Q = +1
    R = 400
    ts = Array(10:10:tmax)
    # ts = round.(Int,range(t0,tmax,length=nsave))

    Δx = NaN*zeros(nsave,R)
    Δy = NaN*zeros(nsave,R)

z=@elapsed Threads.@threads for real in 1:R
    thetas = create_isolated_vortex(L,L,Q)
    omegas = rand(Normal(0,sqrt(Var)),L,L)
    dt = determine_dt(T,Var)

    t = 0.0 ; token_time_t = 1 ; last_loc = r0
    while t < tmax
        t = t + dt
        thetas = update(thetas,omegas,L,T,dt)

        try
            if round(Int,t,RoundDown) ≥ ts[token_time_t]
                last_loc = spot_single_default_local(thetas,last_loc,(Int16.(-1),Int16.(-1)),Q)[1]
                Δx[token_time_t,real],Δy[token_time_t,real] = last_loc .- r0
                token_time_t             = min(token_time_t + 1,nsave)  # not very pretty but here to handle border effects due to rounding issues
            end
        catch e
            if t<tmax println("$real , t = $t : ",e) end
            break
        end
    end
end
println("Runtime = $(round(Int,z)) seconds")

JLD.save("data/Gxt_r$r.jld","Δx",Δx,"Δy",Δy,"Δ",hcat(Δx,Δx))

using StatsBase,LinearAlgebra

T = 0.2
Var = 0.0
# Δx,Δy,Δ,L,tsave = JLD.load("data/Gxt_T$(T)_Var$(Var).jld","Δx","Δy","Δ","L","ts")
Δ,L,tsave = JLD.load("data/Gxt_T$(T)_Var$(Var).jld","Δ","L","ts")
nsave = length(tsave)

pXY=plot(xlabel="|Δx|²/(t/log t)",ylabel="G(|Δx|,t)",title="T = $T, Var = $Var")
    for i in 10:5:nsave
        data = filter(x->!isnan(x),abs.(Δ[i,:]))
        N = length(data)
        h = fit(Histogram,data,0:floor(Int,maximum(data)))
        # h = normalize(h,mode=:density) # if bins not equal
        r = Array(h.edges[1])
        # plot!(r[2:end-1].^2/(tsave[i]./log.(tsave[i])),sqrt(tsave[i]./log(tsave[i]))*remove_negative(h.weights[2:end]/N),rib=0,yaxis=:log,label="t = $(tsave[i])")
        plot!(r[2:end-1].^2/(tsave[i]),sqrt(tsave[i])*remove_negative(h.weights[2:end]/N),rib=0,yaxis=:log,label="t = $(tsave[i])")
        # plot!(r[2:end-1],h.weights[2:end],label="t = $(tsave[i])")
    end
    # plot!(Array(0:0.1:15) .^2,0.3exp.(-Array(0:0.1:15) .^2 / 25),c=:black,label=L"\sim \exp(-x^{2})")
    display(pXY)
savefig("figures\\Gxt_XYb.pdf")
T = 0.05
Var = 0.1
Δ,L,tsave = JLD.load("data/Gxt_T$(T)_Var$(Var).jld","Δ","L","ts")

pNKM=plot(xlabel="|Δx|",ylabel="G(|Δx|,t)",title="T = $T, Var = $Var")
    for i in 5:5:length(tsave)
        data = filter(x->!isnan(x),abs.(Δ[i,:]))
        N = length(data)
        h = fit(Histogram,data,0:floor(Int,maximum(data)))
        # h = normalize(h,mode=:density) # if bins not equal
        r = Array(h.edges[1])
        # plot!(r[2:end-1].^2 /tsave[i]^1.5,1 .*tsave[i]^1.5*remove_negative(h.weights[2:end]/N),rib=0,yaxis=:log)
        plot!(r[2:end-1],remove_negative(h.weights[2:end]/N),rib=0,yaxis=:log)
    end
    display(pNKM)
    ylims!((8E-6,1))
p=plot(pXY,pNKM,layout=2,size=(1200,400))


tsave[5:5:length(tsave)]
pNKMinset=plot(xlabel=L"x^2/t^{3/2}",ylabel=L"t^{3/4}\,G(x,t)",size=(200,200))
    for i in 5:5:length(tsave)
        data = filter(x->!isnan(x),abs.(Δ[i,:]))
        N = length(data)
        h = fit(Histogram,data,0:floor(Int,maximum(data)))
        # h = normalize(h,mode=:density) # if bins not equal
        r = Array(h.edges[1])
        plot!(r[2:end-1].^2 /tsave[i]^1.5,1 .*tsave[i]^0.75*remove_negative(h.weights[2:end]/N),rib=0,yaxis=:log)
    end
    plot!(Array(0.05:0.01:0.4),25exp.(-25Array(0.05:0.01:0.4)),line=:dash,c=:black,yaxis=:log)
    yticks!([1E-3,1E-1,10],[L"10^{-3}",L"10^{-1}",L"10^{1}"])
    xticks!([0,0.1,0.2,0.3,0.4],["0","","","","0.4"])
    display(pNKMinset)
    # ylims!((8E-6,1))
# savefig("figures_draft\\p3ainset.svg")

## Vortex Generation and Visualisation
cd("D:/Documents/Ecole/EPFL/Master_Project/Synchronisation_XY_Model")
using JLD,Dates,Statistics,Distributed,SharedArrays,Distributions,Hungarian
include("methods.jl");
include("methods_vortices.jl");
using Plots,ColorSchemes,LaTeXStrings
gr(box=true,fontfamily="sans-serif",label=nothing,palette=ColorSchemes.tab10.colors[1:10],grid=false,markerstrokewidth=0,linewidth=1.3,thickness_scaling = 1.5)

L         = 200
    Q     = 1       # for a single vortex
    r0    = Int(L/2)    # for a pair of vortices
    T     = 0.05
    Var   = 0.1
    ttr   = Int(500)
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
    @time for i in 1:every:n
        for e in 1:every
            global thetas = update(thetas,omegas,L,T,dt)
            global t += dt
        end
        ~,E = energy(thetas)
        push!(energy_hist,E)
        push!(thetas_hist,thetas)
    end

# files A B C D E Y Z
# T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_Z.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
grille = [(i,j) for i=1:L, j=1:L]  ; xs   = [grille[n][1] for n in 1:length(grille)]  ; ys   = [grille[n][2] for n in 1:length(grille)] ;
smoothedE = (energy_hist[2:end].+abs(minimum(energy_hist[2:end])))
dE = vcat(1,diff(smoothedE))

# plot(Array(1:every:9999)*dt.+ttr,smoothedE[floor(Int,1/every)+1:floor(Int,9999/every)+1])

debut = 1000
fin   = 9000
traj = Matrix{Tuple{Int64,Int64}}(undef,2,floor(Int,(fin-debut+1)/every)+1)
for i in debut:every:fin
    a,b = spot_vortex_antivortex(thetas_hist[floor(Int,i/every)])
    try
        traj[:,floor(Int,(i-debut+1)/every)+1] = [a[1],b[1]]
    catch
        traj[:,floor(Int,(i-debut+1)/every)+1] = [(-1,-1),(-1,-1)]
    end
end

plot((traj[1,:]),c=:black,xlims=(1,L),ylims=(1,L))
plot!((traj[2,:]),c=:black)
z = @elapsed anim = @animate for i in debut:every:fin
    thetas = thetas_hist[floor(Int,i/every)+1]
    pa = plot(Array(debut:every:fin)*dt.+ttr,smoothedE[floor(Int,debut/every)+1:floor(Int,fin/every)+1],size=(1024,1024))
    scatter!((i*dt+ttr,smoothedE[floor(Int,i/every)+1]),m=:star,c=:black,ms=10)


    vortices = spot_vortices(thetas)


    # E_modif = energy(thetas)[1]
    # for vortex in vortices
    #     c,d = vortex[1:2]
    #     for i in c-1:c+1
    #         for j in d-1:d+1
    #             try E_modif[i,j] = -4 catch ; end
    #         end
    #     end
    # end
    #
    # # pb = heatmap(E_modif',size=(1024,1024))
    # pb = heatmap(E_modif',size=(1024,1024),clims=(-4,-2.5))
    # for vortex in vortices
    #     if dE[floor(Int,i/every)+1]>0 mcol = :red else mcol = :green end
    #     if vortex[3]>0 filled = mcol else filled = :transparent end
    #     scatter!((vortex[1:2]).+(0.5,0.5),m=:circle,c=filled,mstrcol=mcol,msc=mcol,msw=2,ms=10)
    # end
    # plot!((traj[1,:]),c=:white)
    # plot!((traj[2,:]),c=:white)
    # xlims!(1,L)
    # ylims!(1,L)


    pc = heatmap(mod.(thetas,2pi)',size=(1024,1024),c=cgrad([:white,:blue,:green,:orange,:red,:white]))
    for vortex in vortices
        if vortex[3]>0 filled = :black else filled = :transparent end
        scatter!((vortex[1:2]).+(0.5,0.5),m=:circle,c=filled,mstrcol=:black,msc=:black,msw=2,ms=10)
    end
    plot!((traj[1,:]),c=:black)
    plot!((traj[2,:]),c=:black)
    xlims!(1,L)
    ylims!(1,L)

    # lay = @layout [a{0.24w} b c]
    plot(pa,pc,layout=2,title="T = $T, Var=$Var, t = $(round(ttr+dt*i,digits=2))",size=(1024*2,1024))
end
println("Time elapsed  = $z seconds")
mp4(anim,"figures_vortices/vizu_vortex_T$(T)_Var$(Var)_test.mp4",fps=10)
&
save("data_vortices\\traj_T$(T)_Var$(Var)_champ.jld","T",T,"Var",Var,"L",L,"ttr",ttr,"dt",dt,"r0",r0,"thetas_hist",thetas_hist,"energy_hist",energy_hist,"n",n,"every",every)


T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_Z.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
    plot(Array(2700:every:9000)*dt.+ttr,smooth(smoothedE[floor(Int,2700/every)+1:floor(Int,9000/every)+1]),xlims=(0,n*dt+ttr))
    scatter!((ttr+dt*5780,smooth(smoothedE[floor(Int,5780/every)])),ms=7,m=:dtriangle)
    scatter!((ttr+dt*7100,smooth(smoothedE[floor(Int,7100/every)])),ms=7,m=:utriangle)

    energyy_1 = energy(thetas_hist[floor(Int,5780/every)+1])[1]
    energyy_1 .+= 4 # E_0
    heatmap(energyy_1',clims=(0,1.))

    energyy_2 = energy(thetas_hist[floor(Int,7200/every)+1])[1]
    energyy_2 .+= 4
    heatmap(energyy_2',clims=(0,1))


T,Var,L,ttr,dt,r0,thetas_hist,energy_hist,n,every = load("data_vortices\\energy_Y.jld","T","Var","L","ttr","dt","r0","thetas_hist","energy_hist","n","every")
    plot(Array(2000:every:9999)*dt.+ttr,smooth(smoothedE[floor(Int,2000/every)+1:floor(Int,9999/every)+1]),xlims=(0,n*dt+ttr))
    scatter!((ttr+dt*5200,smooth(smoothedE[floor(Int,5200/every)])),ms=7,m=:dtriangle)
    scatter!((ttr+dt*6600,smooth(smoothedE[floor(Int,6600/every)])),ms=7,m=:utriangle)

    energyy_1 = energy(thetas_hist[floor(Int,5200/every)+1])[1]
    energyy_1 .+= 4
    heatmap(energyy_1',clims=(0,1.))

    energyy_2 = energy(thetas_hist[floor(Int,6600/every)+1])[1]
    energyy_2 .+= 4
    heatmap(energyy_2',clims=(0,1))
