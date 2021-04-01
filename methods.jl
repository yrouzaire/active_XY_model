"Copyright (c) 2021 Y.Rouzaire All Rights Reserved."

## Methods to deal with a square lattice LxL with periodic BCs
@everywhere function dist(a::Tuple{Int,Int},b::Tuple{Int,Int},L::Int)::Float64  # euclidian distance with Periodic BCs
    dx = abs(a[1] - b[1]) ; dx = min(dx,L-dx)
    dy = abs(a[2] - b[2]) ; dy = min(dy,L-dy)
    return sqrt(dx^2 + dy^2)
end

@everywhere function linear_to_square_index(n::Int,L::Int)
    i = div(n-1,L) + 1  # 1 ≤ i ≤ L
    j = mod1(n,L)       # 1 ≤ j ≤ L
    #= formula for reversing the linear indexing. i is the quotient and j
    the reminder of the euclidian division of n by L and the corrections
    deal with the 1-indexing of Julia =#
    return i,j
end

@everywhere function square_to_linear_index(i::Int,j::Int,L::Int)::Int
    return L*(i-1) + j # 1 ≤ n ≤ L^2
end

@everywhere function get_neighbours(L::Int,i::Int,j::Int)::Array{Tuple{Int64,Int64},1}
    return [(mod1(i+1,L),j) , (mod1(i-1,L),j) , (i,mod1(j+1,L)) , (i,mod1(j-1,L))]
end

## Creation and Saving Methods
@everywhere function create_Lattice(L,T,Var_omega,typeof_init)::Tuple{Array{Float64,3},Float64} # creates a completely new lattice
    lattice = Array{Float64,3}(undef,L,L,2)

    # initialisation of angles (by convention, angle = first element of the 3rd dimension)
    if typeof_init == "LowTemp"
        lattice[:,:,1] = zeros(L,L)
    elseif typeof_init == "HighTemp"
        lattice[:,:,1] = rand(L,L)*2π
    elseif typeof_init == "Half" # first half up, second half down
        L_over2 = round(Int,L/2)
        lattice[1:L_over2,:,1] = zeros(L_over2,L)
        lattice[L_over2+1:L,:,1] = ones(L-L_over2,L)*π
        lattice[1:L_over2,:,2] = zeros(L_over2,L)
        lattice[L_over2+1:L,:,2] = ones(L-L_over2,L)*π
    else error("ERROR : Type of initialisation unknown. Choose among \"HighTemp\",\"LowTemp\" or \"Half\" .")
    end

    # initialisation of intrinsic frequencies (by convention, omega = second element of the 3rd dimension)
        # WARNING : in Julia, "Normal(0,σ)" delivers a N(0,σ²) Gaussian distribution
    σ = sqrt(Var_omega)
    lattice[:,:,2] = rand(Normal(0,σ),L,L)

    # Timestep computation
    arbitrary_coeff =  π/10
    dt = min(arbitrary_coeff/maximum(abs.(lattice[:,:,2])) , arbitrary_coeff/2 , arbitrary_coeff^2*π/4/T)
    #= A few remarks on the previous line :
        1. Julia can deal with 1/0 = Inf and any number is smaller than Inf
            so no problem with the divisions
        2. The first term (π/10max) is here to ensure that the oscillator
            with the max frequency will be resolved correctly. It grows as sqrt(logN).
        3. The second term should have been π/40 to ensure that the greater
            possible coupling (when each one of all 4 neighbours produces a
            unit coupling). Since this situation only happens at very high T°,
            that it will basically never occur, I chose to relax this upper bound
            to π/20. It's more of a guardrail in case of very small σ AND very
            small T.
        4. The last term is obtained by getting the expectation value of the
            absolute value of the noise ξ, leading to sqrt(π/T)/2. Remember that
            it is sqrt(dt) and not dt alone that enters the Euler Maryuama integrator.
            It is the more constraining term for usual ranges of \sigma and T.
        5. The overall coefficient π/10 is arbitrary and can be modulated as desired.
        =#

    return lattice , dt
end

@everywhere function create_Lattice_from_Realisation(filename)::Tuple{Array{Float64,3},Float64,Float64,Float64,Float64} # reads everything from a JLD file
    data    = load(filename)
    lattice = data["lattice"]
    K       = data["K"]
    sigma2  = data["Var_omega"]
    dt      = data["dt"]
    T       = data["T"]
    return lattice,K,sigma2,dt,T
end

@everywhere function save_system(lattice,dt,tmax,K,T,Var_omega)
    str = "lattice_"*string(Dates.day(now()))
    JLD.save("data/"*str*".jld", "lattice", lattice,"K",K,"T",T,"Var_omega",Var_omega,"dt",dt)
end

## Get info about nearest spins
@everywhere function get_angle_nearest_neighbours(thetas::Matrix{Float64},i::Int,j::Int,L::Int)::Vector{Float64}
    return [thetas[mod1(i+1,L),j], ## Note : mod1 instead of mod to deal with the indexing at from 1
            thetas[mod1(i-1,L),j],
            thetas[i,mod1(j+1,L)],
            thetas[i,mod1(j-1,L)] ]
end

@everywhere function get_omega_nearest_neighbours(omegas::Matrix{Float64},i::Int,j::Int,L::Int)::Vector{Float64}
    return [omegas[mod1(i+1,L),j], ## Note : mod1 instead of mod to deal with the indexing at from 1
            omegas[mod1(i-1,L),j],
            omegas[i,mod1(j+1,L)],
            omegas[i,mod1(j-1,L)] ]
end

## "Get" methods
@everywhere function get_Delta(thetas::Matrix{Float64})::Vector{Float64}
    L = size(thetas)[1]
    Δ = zero(ComplexF64)
    Δ_proj = 0.0 # Δ projected on x-axis
    for theta in thetas # scan all spin values, no matter the order
        Δ      += exp(im*theta) # im is the imaginary unit : im² = -1
        Δ_proj += cos(theta)
    end
    Δ = abs(Δ)/L^2
    Δ_proj = Δ_proj/L^2
    return [Δ,Δ_proj]
end


@everywhere function get_delta(thetas::Matrix{Float64},L::Int,l::Int)::Matrix{Float64}
    # L is the size of the system
    # l is the size of the local box for averaging the magnetisation
    @assert isinteger(L/l)
    c = Int(L/l)
    δ = zeros(2,c^2)
    for j in 1:c
        for i in 1:c
            δ[:,square_to_linear_index(i,j,c)] = get_Delta(thetas[1+l*(i-1):i*l, 1+l*(j-1):j*l])
        end
    end
    return δ
end

@everywhere function get_C_perp(thetas::Matrix{Float64},i::Int,j::Int,n::Int,L::Int)::Float64 # used in get_C below
    angle_neighbours_at_distance_n = [thetas[mod1(i+n,L),j], ## Note : mod1 instead of mod to deal with the indexing at from 1
                        thetas[mod1(i-n,L),j],
                        thetas[i,mod1(j+n,L)],
                        thetas[i,mod1(j-n,L)] ]
    return mean(Float64[cos(angle - thetas[i,j]) for angle in angle_neighbours_at_distance_n])
end

@everywhere function get_C_diag(thetas::Matrix{Float64},i::Int,j::Int,n::Int,L::Int)::Float64 # used in get_C below
    angle_neighbours_at_distance_sqrt2_n = [thetas[mod1(i+n,L),mod1(j+n,L)], ## Note : mod1 instead of mod to deal with the indexing at from 1
                        thetas[mod1(i+n,L),mod1(j-n,L)],
                        thetas[mod1(i-n,L),mod1(j+n,L)],
                        thetas[mod1(i-n,L),mod1(j-n,L)] ]
    return mean(Float64[cos(angle - thetas[i,j]) for angle in angle_neighbours_at_distance_sqrt2_n])
end

@everywhere function get_C(thetas::Matrix{Float64})::Vector{Float64} # returns C(r) for angles
    L = size(thetas)[1] ; Lover2 = round(Int,L/2,RoundDown)
    matrix = Matrix{Float64}(undef,L,L^2)

    # r_vector = vcat([n for n in 1:round(Int,L/2,RoundDown)],[n*sqrt(2) for n in 1:round(Int,L/2,RoundDown)])

    # We fill the matrix with perpendicular directions first (vertical and horizontal)
    for i in 1:L
        for j in 1:L
            for n in 1:Lover2
                matrix[n,L*(i-1) + j] = get_C_perp(thetas,i,j,n,L)
                # (i,j) -> L*(i-1) + j in linear indexing
            end
        end
    end

    # We then fill the matrix with diagonal directions
    for i in 1:L
        for j in 1:L
            for n in 1:Lover2
                matrix[n+Lover2,L*(i-1) + j] = get_C_diag(thetas,i,j,n,L)
            end
        end
    end

    result_unsorted = Statistics.mean(matrix,dims=2) # average over spins = over columns
        # discretization of the distance r, in 2 parts for simplicity
        # in the matrix construction
    r_vector = vcat([n for n in 1:Lover2],[n*sqrt(2) for n in 1:Lover2])
    sortpermm = sortperm(r_vector)
    result_sorted = result_unsorted[sortpermm]

    return result_sorted # finally, one gets C(t,r)

    #= Note 1 : Since the work needed to fill each cell of the Matrix "matrix"
    is consequent, I believe the ordering row vs column will not be limiting
    in this case. =#

    #= Note 2 : This part of the code is of complexity O(L^3). It is already
    one order of complexity above the other methods, which are usually O(L^2).
    This is why I chose not to implement other approximations for the euclidian
    distance r, which would have pleasantly refined its discretization.
    There are a lot of other possibilities : i.e. the displacement of a cavalier
    in a chess game. But already there, the combinatorics begin to increase
    (8 possibilities). Implementing more and more of these would let the complexity
    go up to  O(L^4). =#

    #= Note 3 : One may remark that the output is an average of averages. In this
    particular case, it is totally correct because all the sub-samples are of the
    same size. Note that this is another argument is favour of not computing other
    types of discretized distances. =#
end

@everywhere function C_all_directions(lattice) # to make sure there is no anisotropy
    L = size(lattice)[1] ; L2 = round(Int,L/2)
    C_vertical = zeros(L2,L^2)
    C_horizontal = zeros(L2,L^2)
    C_diag = zeros(L2,L^2)
    C_anti_diag = zeros(L2,L^2)
    for n in 1:1:L^2
        i,j = linear_to_square_index(n,L)
        angle = lattice[i,j,1]
        for k in 1:L2
            C_vertical[k,n]   = cos(angle - lattice[mod1(i+k,L),j,1])
            C_horizontal[k,n] = cos(angle - lattice[i,mod1(j+k,L),1])
            C_diag[k,n]       = cos(angle - lattice[mod1(i+k,L),mod1(j+k,L),1])
            C_anti_diag[k,n]  = cos(angle - lattice[mod1(i-k,L),mod1(j+k,L),1])
        end
    end
    C_vertical = mean(C_vertical,dims=2)
    C_horizontal = mean(C_horizontal,dims=2)
    C_diag = mean(C_diag,dims=2)
    C_anti_diag = mean(C_anti_diag,dims=2)

    return C_vertical , C_horizontal , C_diag , C_anti_diag
end


@everywhere function get_Ctw(thetas_t::Matrix{Float64},t::Float64,token_time_t::Int,lattices_tw::Array{Matrix{Float32},1},tws::Vector{Int64},Ctw::Matrix{Float64}) ## remplit une colonne de la matrice Ctw
    for i in eachindex(tws)
        tw = tws[i]
        if t > tw # on peut remplir la colonne 'token_time_t' de la matrice Ctw
            Ctw[i,token_time_t] = mean(cos.(thetas_t .- lattices_tw[i])) # returns ⟨∑ cos(θ_i(t) - θ_i(t_w))⟩/L²
        else
            #= On ne peut pas encore, donc on écrit des NaN.
            Missing aurait été plus approprié mais ce n'est pas un Float, et on obtient
            l'erreur suivante : ArgumentError: type of SharedArray elements must
            be bits types, got Union{Missing, Float64} =#
            Ctw[i,token_time_t] = NaN
        end
    end
    return nothing # one writes the measurement in place, in the matrix Ctw
end


@everywhere function get_dispersion_theta(lattice0::Matrix{Float64},lattice_final::Matrix{Float64})::Array{Float64,2}
    #= Inputs :
    lattice0 = lattice at time t0, at equilibrium
    lattice_final = lattice at final time tmax =#
    return (lattice_final .- lattice0) .^2
end


@everywhere function get_fastest_oscillators(omegas::Matrix{Float64} ; threshold::Number = NaN , number::Int = 0 )
    threshold_is_modified = threshold ≠ NaN && threshold > 0
    number_is_modified    = threshold ≠ 0   && number    > 0
    @assert xor(threshold_is_modified,number_is_modified) "Either the entered value is incorrect or both keyargs were used at the same time."
    L = size(omegas)[1]
    if threshold_is_modified
        result = Tuple{Int,Int}[]
        for j in 1:L
            for i in 1:L
                if abs.(omegas[i,j]) > threshold push!(result,(i,j)) end
            end
        end
        return result
    elseif number_is_modified
        omegas_abs = abs.(omegas)
        return linear_to_square_index.(Array(partialsortperm(vec(omegas_abs), 1:number, rev=true)),L)
    end
end

## Gestion des Runaways
@everywhere function spot_runaways(lattice_omega::Matrix{Float64},threshold::Number,nbroken::Int)::Vector{Tuple{Int,Int}} # return their location (i,j) in a Array of Tuple
    runaway_locations = Tuple[]
    L = size(lattice_omega)[1]

    for i in 1:L
        for j in 1:L
            omega_spin = lattice_omega[i,j]
            omega_neighbours = get_omega_nearest_neighbours(lattice_omega,i,j,L)
            Δω = abs.(omega_neighbours .- omega_spin)

            # if sum(Δω .> threshold) ≥ nbroken
            if sum(Δω .> threshold) == nbroken
                push!(runaway_locations,(i,j))
            end
        end
    end
    return runaway_locations
end

@everywhere function create_lattice_runaway(L,T,Var,init,threshold,nbroken)::Tuple{Array{Float64,3},Float64,Int,Int} # return lattice,dt,i_runaway,j_runaway
    if Var == 0 error("σ = 0 , no runaway configuration is possible !") end
    ntries_max = 10000 ; ntries = 1
    while ntries < ntries_max
        try
        # Create a lattice
        lattice,dt = create_Lattice(L,T,Var,init)
        current_nbroken = nbroken # the nbroken at which one currently works (will decrease in the SEARCH PHASE and increase in the SWAP PHASE)

        ## SEARCH PHASE : Find the weakest spin (the one that has the most broken links)
        iweak,jweak = -1,-1 # will contain the location of the runaway
        omegaweak = NaN      # will contain the \omega of the runaway
        list_runaways = spot_runaways(lattice[:,:,2],threshold,current_nbroken) # array of tuples (i,j)
        if length(list_runaways) > 0
            iweak,jweak = list_runaways[1][1],list_runaways[1][2]
            return lattice,dt,iweak,jweak # because the lattice has already what you want, i.e at least one runaway oscillator (thus return the first element of the list)
        else # there is currently no runaways with the demanded threshold/nbroken
            bool_found = false
            current_nbroken = current_nbroken - 1 # now we look for runaways in the unchanged lattice but with a looser constraint : nbroken-1
            while current_nbroken > 0
                list_runaways = spot_runaways(lattice[:,:,2],threshold,current_nbroken)
                if length(list_runaways) > 0
                    #= we have a list of tuples of weak spins. We want the most fragile,
                    i.e. the one that has the largest intrinsic frequency difference
                    with its neighbours involved in currently existing links =#
                    tmp = 0 # will contain the index of the weakest runaway found so far
                    value_tmp = -1 # will contain the biggest frequency difference found so far
                    for i in eachindex(list_runaways) # scan over all runaways
                        # for each runaway, get the sum of the frequency difference with their existing neighbours
                        omega_neighbours = [ lattice[neighbour[1],neighbour[2],2] for neighbour in get_existing_neighbours(lattice,list_runaways[i][1],list_runaways[i][2],threshold) ]
                        diff_omega_neighbours = abs.( omega_neighbours .- lattice[list_runaways[i][1],list_runaways[i][2],2] )
                        sum_diff_frequency = sum(diff_omega_neighbours)
                        if sum_diff_frequency > value_tmp
                            tmp = i
                        end
                    end
                    # We now have the most fragile of the spins :
                    iweak,jweak = list_runaways[tmp][1] , list_runaways[tmp][2]
                    omegaweak   = lattice[iweak,jweak,2]
                    bool_found  = true
                    break # we found the weakest spin -> phase 1 (SEARCH) is thus over. One goes directly to PHASE 2 : the swap procedure
                else
                    current_nbroken = current_nbroken - 1 # now we look again for runaways in the unchanged lattice but with a looser constraint : nbroken-1
                end # if
            end # while current_nbroken > 0
            if !bool_found error("No runaway with nbroken ≥ 1 ! Try again.") end # If one gets here, it means that there is no runaway with nbroken ≥ 1
        end # if length() > 0

            ## SWAP PHASE
            #= at this stage, "weak_spin" has only current_nbroken links broken.
            We aim for nbroken links. Thus, we are going to swap spins involved
            in existing links to weaken a link so that it breaks. Therefore, the
            number of actually broken links will now increase. =#
            broken_neighbours_list = get_broken_neighbours(lattice,iweak,jweak,threshold) # list of tuples (i,j) of neighbours of the weak spin that are involved in a broken link
            while current_nbroken < nbroken
                i_existing_neighbour,j_existing_neighbour = get_existing_neighbours(lattice,iweak,jweak,threshold)[1] # one gets the/one neighbour (of our weak spin) that is involved in a still existing link

                # Let's find the spin most distant to the weak spin's omega
                mdsf = (0,0,0) # will contain both the coordinates and Δω of the Most Distant (wrt ω) So Far (=mdsf) to our weak spin throughout the search : (i,j,Δω)
                for i in 1:L
                    for j in 1:L
                        Δω = abs(lattice[i,j,2] - omegaweak)
                        if Δω > mdsf[3] && (i,j) ∉ broken_neighbours_list # check whether its not one of the already broken neighbours
                            mdsf = (i,j,Δω) # update the Most Distant spin So Far in our search
                        end
                    end
                end

                # Check whether a swap would break the link. If so, proceed to the swap. If not, throw an error and abort mission.
                if abs(omegaweak - lattice[mdsf[1],mdsf[2],2]) ≥ threshold # a swap would break the new link
                    # swap both spins
                        tmp = lattice[mdsf[1],mdsf[2],:]
                        lattice[mdsf[1],mdsf[2],:] = lattice[i_existing_neighbour,j_existing_neighbour,:]
                        lattice[i_existing_neighbour,j_existing_neighbour,:] = tmp
                    current_nbroken = current_nbroken + 1 # the swap has broken the link ...
                    push!(broken_neighbours_list,(i_existing_neighbour,j_existing_neighbour)) # ... so one has to add it to the list of broken neighbours
                    # from there, the while loop will continue and if the required number of broken neighbours is met, job is done. Otherwise, loop again
                else
                    # even a swap with the furthest spin would not break the link, our quest is hopeless
                    error("There is no spin in this lattice distant enough to meet the requierements in terms of threshold/nbroken. Try again")
                end
            end # While current_nbroken < nbroken

            # if you made it until here, it's because the modified lattice now has what you want : a runaway oscillator at location (iweak,jweak)
            return lattice,dt,iweak,jweak
        catch
            ntries = ntries + 1
        end # end try
    end # while try delivers an error
    error("After $ntries_max attempts, it has been impossible to create the requiered lattice.")
end

@everywhere function get_broken_neighbours(lattice,iweak,jweak,threshold)::Vector{Tuple{Int,Int}} # returns a list of tuples (i,j) of neighbours of the weak spin that are involved in a broken link
    L = size(lattice)[1]
    list = Tuple[]
    neighbours = get_neighbours(L,iweak,jweak) # neighbours is a list of tuples (i,j)
    for neighbour in neighbours
        i_neighbour = neighbour[1] ; j_neighbour = neighbour[2]
        if abs(lattice[iweak,jweak,2] - lattice[i_neighbour,j_neighbour,2]) ≥ threshold # that link is broken
            push!(list,neighbour)
        end
    end
    # return is hidden in next line
    if length(list) > 0 return list else error("No broken neighbours. Something must have gone wrong.") end
end

@everywhere function get_existing_neighbours(lattice,iweak,jweak,threshold)::Vector{Tuple{Int,Int}}  # returns all neighbours of the weak spin that are involved in an existing link.
    L = size(lattice)[1]
    list = Tuple[]
    neighbours = get_neighbours(L,iweak,jweak) # neighbours is a list of tuples (i,j)
    for neighbour in neighbours
        i_neighbour = neighbour[1] ; j_neighbour = neighbour[2]
        if abs(lattice[iweak,jweak,2] - lattice[i_neighbour,j_neighbour,2]) < threshold # that link does exists
            push!(list,neighbour)
        end
    end

    # return is hidden in next line
    if length(list) > 0 return list else error("No effective (contrary of 'broken') neighbours. Something must have gone wrong.") end
end

## Gestion des Vortex
@everywhere function arclength(theta1::Float64,theta2::Float64)::Float64
    #= This function returns the signed arclength on the unit trigonometric circle .
    Clockwise        >> sign -
    Counterclockwise >> sign +
    Note that the inputs thetas are within [0,2π] =#
    dtheta = theta2 - theta1
    dtheta_abs = abs(theta2 - theta1)

    shortest_unsigned_arclength = min(2π-dtheta_abs,dtheta_abs)
    if dtheta_abs < π
        signe = sign(dtheta)
    else
        signe = -sign(dtheta)
    end
    return signe*shortest_unsigned_arclength
end

@everywhere function get_vorticity(thetas::Matrix{Float64},i::Int,j::Int,L::Int)::Int
    #= Note : here, i and j are the coordinates of the plaquette one considers.
        The top-left corner spin of a plaquette (i,j) also has coordinates (i,j).
        By convention, a plaquette has an orientation : we start with the topleft
        corner and rotate in the Counterclockwise direction, hence the particular
        ordering of the spins in the following line of code
        1---4
        |   |
        2---3
        =#
    angles_corners = mod.([thetas[i,j],thetas[mod1(i+1,L),j],thetas[mod1(i+1,L),mod1(j+1,L)],thetas[i,mod1(j+1,L)]],2π)
    perimeter_covered  = arclength(angles_corners[1],angles_corners[2])
    perimeter_covered += arclength(angles_corners[2],angles_corners[3])
    perimeter_covered += arclength(angles_corners[3],angles_corners[4])
    perimeter_covered += arclength(angles_corners[4],angles_corners[1])
    # if isnan(perimeter_covered) println("i $i j $j lattice $lattice_angles") end
    return round(Int,perimeter_covered/2π)
end

@everywhere function spot_vortices(thetas::Matrix{Float64})::Vector{Tuple{Int,Int,Int}}
    L = size(thetas)[1]
    list_vortices = Tuple{Int,Int,Int}[] # will contain a tuple for each vortex : (i,j,q), where q is its charge
    for i in 1:L
        for j in 1:L
            q = get_vorticity(thetas,i,j,L)
            if q ≠ 0
                push!(list_vortices,(i,j,q))
            end
        end
    end
    # @assert sum_q == 0
    return list_vortices
end

@everywhere function spot_vortex_antivortex(thetas::Matrix{Float64})
    L = size(thetas)[1]
    list_vortex     = Tuple{Int,Int}[] # will contain a tuple for each vortex     : (i,j)
    list_antivortex = Tuple{Int,Int}[] # will contain a tuple for each antivortex : (i,j)
    for i in 1:L
        for j in 1:L
            q = get_vorticity(thetas,i,j,L)
            if     q == +1 push!(list_vortex,(i,j))
            elseif q == -1 push!(list_antivortex,(i,j))
            end
        end
    end
    return list_vortex,list_antivortex
end

@everywhere function number_vortices(thetas::Matrix{Float64})
    L = size(thetas)[1]
    nb_vortices = 0
    for i in 1:L
        for j in 1:L
            if get_vorticity(thetas,i,j,L) ≠ 0
                nb_vortices = nb_vortices + 1
            end
        end
    end
    return nb_vortices
end

@everywhere function classify_vortices(thetas::Matrix{Float64})
    #= Let's sum up what this function does. Characterising a vortex pair as free
    or bounded is not a trivial question, be it numerically or conceptually.
    There currently exists no precise definition in the litterature and what follows
    does not pretend to be a definitive one. The underlying idea is to OPTIMALLY
    pair the vortices with their closest opposite-polarity neighbour, optimal in
    the sense that the total distance between pairs should be minimized.
    Then, we define a pair as free if its distance is above a given threshold,
    and as bounded if below that threshold. If one decides to reason in terms of
    distances to classify pairs of vortices, I reckon this optimal bipartite matching
    is the only valid solution. Any of the other simpler procedures I've tried led to
    incoherences, at least when visually checking the results.

    To carry out the bipartite optimal matching, we use the so-called Hungarian Matching algoritm,
    also called Munkres' algoritm after James Munkres, the first to have proved that the
    complexity of this algo was not actually exponential as thought until then, but rather
    polynomial : O(m^3) where m is the number of vortex pairs. (for the record, he showed
    it was O(m^4) and the bound was lowered in the 70's).
    Note 1 : m = O(L²) is the worst-case (high T and/or high σ²), leading the algorithm
        to be O(L^6) in the worst-case. =#

    L = size(thetas)[1]

    list_vortex,list_antivortex  = spot_vortex_antivortex(thetas) # Vector{Tuple{Int,Int}}
    m = length(list_vortex) # number of pairs vortex/antivortex

    # Labels
        # label = 0  : free vortex
        # label ≥ 1  : bounded vortex, each pair has a unique ID (the token) ≥ 1
    label_vortex = fill(-1,m) ; label_antivortex = fill(-1,m)
    token = 1

    #= Now comes the tricky question of what distance should we choose as
    threshold to determine whether a vortex pair is bounded or not. Let's dare the
    crude approximation that the pairs of vortices are uniformly distributed on
    the square.
    Let's consider the behaviour only at large L (actually, I reckon that this second
    approximation should be more controlled in a periodic lattice).
    Based on the first answer by Douglas Zare on mathoverflow.net/questions/124579
    the average minimal distance between each one of the m/2 pairs is of the order of 2/m.
    Since if the threshold is too large, one will never end up with free vortices
    classified as such, let's divide for security this threshold by a factor two,
    leading to threshold_dist = 1/m (here for a unit square -> L/m for a LxL lattice.)

    Note 2 : the threshold distance decreases exponentially with T and/or σ² since
        the number of vortices increases exponentially.
    Note 3 : since m = O(L²) is the worst-case, one has to manually ensure that
        the threshold does not decrease too much. Here comes in the first arbitrary
        parameter, namely that constant in the max. Indeed, we observe in the simulations
        that vortices that one would intuitively consider as bounded can very well diffuse
        (on distance of the order of a few lattice-spacings) due to random thermal
        fluctuations.
        The value 3 is justified thanks to the benchmarks, showing that for ≤ 2 and ≥ 5
        it produces poor results (L=200 , R=30 , tmax=500) =#
    threshold_dist = max(3,L/m)

    dist_matrix = Array{Float64,2}(undef,m,m)
    for j in 1:m # j is an antivortex
        for i in 1:m # i is a vortex
            dist_matrix[i,j] = dist(list_vortex[i],list_antivortex[j],L)
        end
    end

    try # if there is no vortex at all, hungarian rises "ArgumentError: reducing over an empty collection is not allowed"
        matching,~ = Hungarian.hungarian(dist_matrix) # here is done the optimal bipartite matching

        for i in 1:m # i is a vortex
            j = matching[i] # j is an antivortex
            if dist_matrix[i,j] ≤ threshold_dist # you just found its pair, so label it with a unique token !
                label_vortex[i] = token
                label_antivortex[j] = token
                token += 1
            else # its closest pair is further than the threshold -> free vortex
                label_vortex[i] = label_antivortex[j] = 0
            end
        end
        number_free_vortices  = 2sum(label_vortex .== 0)
        number_all_vortices   = 2m

        return label_vortex,label_antivortex,number_all_vortices,number_free_vortices
    catch e
        # println("Caught Error ",sprint(showerror, e))
        return Tuple{Int,Int}[],Tuple{Int,Int}[],0,0  # if there was no vortex at all in the first place, output this
    end
end

## Time evolution methods
@everywhere function update(lattice::Array{Float64,3},T::Number,L::Int,dt::Float64)::Array{Float64,3}
    noise = randn(L,L)
    thetas = lattice[:,:,1]
    omegas = lattice[:,:,2]
    thetas_new = Array{Float64,2}(undef,L,L)
    for j in 1:L
        for i in 1:L
            #= NB : the inversion j,i is wanted. Because Julia accesses the matrices by columns and not by rows,
            for i,j leads to O(L^3) while for j,i leads to O(L^2), as expected.  =#
            angle_neighbours = get_angle_nearest_neighbours(thetas,i,j,L) # O(1)
            θ,ω = thetas[i,j] , omegas[i,j]
            sin_angle_neighbours = sin.(angle_neighbours .- θ)
            thetas_new[i,j] =  θ + ω*dt + dt*sum(sin_angle_neighbours) + sqrt(2*T*dt)*noise[i,j] # O(1)
        end
    end
    return cat(thetas_new , omegas , dims=3)
end

@everywhere function evolve(lattice::Array{Float64,3},l::Int,T::Number,dt::Float64,tmax::Int,tsave,tws)
    L    = size(lattice)[1] ; nsave = length(tsave)

    Δ    = Matrix{Float64}(undef,2,nsave)             # magnetisation
    δ    = Array{Float64,3}(undef,2,Int(L/l)^2,nsave)  # local magnetisation
    C    = Matrix{Float16}(undef,L,nsave)           # correlation function C(r)
    Ctw  = Matrix{Float64}(undef,length(tws),nsave) # correlation function C(t,tw)
    nb_all_vortex  = Vector{Int}(undef,nsave)
    nb_free_vortex = Vector{Int}(undef,nsave)

    lattices_tw = Vector{Matrix{Float32}}(undef,length(tws))
    # lattices_t  = Matrix{Float32}(undef,L^2,nsave)

    t = 0.0 ; token_time_t = 1 ; token_time_tw = 1
    while t < tmax
        t = t + dt
        lattice = update(lattice,T,L,dt)

    # save lattices at tw
        # if token_time_tw ≤ length(tws) round(Int,t,RoundDown) ≥ tws[token_time_tw]
        #     lattices_tw[token_time_tw] = lattice[:,:,1]
        #     token_time_tw += 1
        # end

    # save quantities of interest
        if  round(Int,t,RoundDown) ≥ tsave[token_time_t]
            thetas = lattice[:,:,1]
            # Δ[:,token_time_t]        = get_Delta(thetas)            # O(L^2)
            # δ[:,:,token_time_t]      = get_delta(thetas,L,l)        # O(L^2)
            C[:,token_time_t]        = get_C(thetas)                # O(L^3)
            # get_Ctw(thetas,t,token_time_t,lattices_tw,tws,Ctw)      # O(L^2) , in-place operation (modifies directly Ctw)
            # ~,~,nb_all_vortex[token_time_t],nb_free_vortex[token_time_t] = classify_vortices(thetas)     # O(L^6) worst case ; O(L^2) best case
            # lattices_t[:,token_time_t] = vec(thetas)
            token_time_t             = min(token_time_t + 1,nsave)  # not very pretty but here to handle border effects due to rounding issues
        end
    end

    return Δ,δ,C,Ctw,nb_all_vortex,nb_free_vortex
end

## Meta Methods
@everywhere function scan(configfile_name="configuration.jl")
    include(configfile_name)
    include("IDrealisation.jl")
    len = length(TV)

    # Declarations of the output containers
    Δ   = SharedArray{Float64,3}(len,2,nsave) # global magnetisation on the LXL system (genuine and projected on x-axis)
    δ   = SharedArray{Float64,4}(len,2,Int(L/l)^2,nsave) # local magnetisation on a lxl domain (genuine and projected on x-axis)
    C   = SharedArray{Float16,3}(len,L,nsave)
    Ctw = SharedArray{Float64,3}(len,length(tws),nsave)
    number_all_vortex  = SharedArray{Int,2}(len,nsave)
    number_free_vortex = SharedArray{Int,2}(len,nsave)
    # thetas_t           = SharedArray{Float16,3}(len,L^2,nsave) # vec(thetas) for each time saved

    LS = fill(L,len) ; lS = fill(l,len) ; initS = fill(init,len) ; tsaveS = fill(tsave,len) ; tmaxS = fill(tmax,len) ; twsS = fill(tws,len)
    ΔS = fill(Δ,len) ; δs = fill(δ,len) ; CS = fill(C,len) ; CtwS = fill(Ctw,len) ; number_all_vortexS = fill(number_all_vortex,len) ; number_free_vortexS = fill(number_free_vortex,len)
    pmap(simulation,LS,lS,TV,initS,tmaxS,tsaveS,twsS,Array(1:len),ΔS,δs,CS,CtwS,number_all_vortexS,number_free_vortexS)

    nT,nVar = length(Ts),length(Vars)
    # Convert the containers to the right format
    Δ   = convert(Array,reshape(Δ,nT,nVar,2,nsave))
    δ   = convert(Array,reshape(δ,nT,nVar,2,Int(L/l)^2,nsave))
    C   = convert(Array,reshape(C,nT,nVar,L,nsave))
    Ctw = convert(Array,reshape(Ctw,nT,nVar,length(tws),nsave))
    number_all_vortex  = convert(Array,reshape(number_all_vortex,nT,nVar,nsave))
    number_free_vortex = convert(Array,reshape(number_free_vortex,nT,nVar,nsave))
    # thetas_t           = convert(Array,reshape(thetas_t,nT,nVar,L^2,nsave))

    ## Save Outputs
    # println("On y arrive.")
    filename = "data/"*fname*"_r$r.jld"
    JLD.save(filename,"Δ",Δ,"δ",δ,"C",C,"Ctw",Ctw,"number_all_vortex",number_all_vortex,"number_free_vortex",number_free_vortex,"L",L,"l",l,"Vars",Vars,"Ts",Ts,"init",init,"tmax",tmax,"tsave",tsave,"tws",tws,"name",name,"comments",comments)
    println("Data saved in ",filename)
end


@everywhere function simulation(L,l,TV,init,tmax,tsave,tws,i,Δ,δ,C,Ctw,number_all_vortex,number_free_vortex)
    T,Var = TV
    println("Simulation at T=$T & σ²=$Var launched at $(Dates.hour(now())):$(Dates.minute(now()))")
    lattice,dt = create_Lattice(L,T,Var,init)
    Δ[i,:,:],δ[i,:,:,:],C[i,:,:],Ctw[i,:,:],number_all_vortex[i,:],number_free_vortex[i,:] = evolve(lattice,l,T,dt,tmax,tsave,tws)
end

## Visualisation methods
# @everywhere function smooth(X) ## for smoother plots
#     smoothed = copy(X)
#     coeff = [1,2,1]
#     coeff = coeff./sum(coeff)
#     # @assert isodd(length(coeff))
#     s = Int(floor(length(coeff)/2))
#     for i in 1+s:length(smoothed)-s
#         smoothed[i] = X[i-s:i+s]'*coeff
#     end
#     return smoothed
# end

# @everywhere function smooth(X;over=3) ## for smoother plots
#     smoothed = copy(X)
#     coeff = [2^(i-1) for i in 1:over]
#     coeff = coeff./sum(coeff)
#     s = length(coeff)
#     for i in 1+s:length(smoothed)
#         smoothed[i] = X[i-s+1:i]'*coeff
#     end
#     return smoothed, coeff
# end

@everywhere function remove_negative(series)
    result = Vector{Union{Float64,Missing}}(undef,length(series)) # initializes by default to "missing"
    for i in eachindex(series)
        if series[i] > 0
            result[i] = series[i]
        end
    end
    return result
end

@everywhere function correct_location(x::Tuple{Int,Int},L::Int)::Tuple{Int,Int}
    #= This method deals with the difference in ordering between the lattice of
        angles and the way plots are layed out. Thetas spans from (1,1) in top left
        corner to (L,L) at bottom right. Plots layout goes from (1,1) in bottom left to (L,L) at
        top right. Thus, one needs to invert rows&columns and to reverse the order of the (new)
        columns (going down instead of going up).
        One can find these explanations with schemes at page 21 of the handwritten book. =#
    return (x[2],L-x[1])
end
#
# @everywhere function twiny(sp::Plots.Subplot)
#     sp[:top_margin] = max(sp[:top_margin], 40Plots.px)
#     plot!(sp.plt, inset = (sp[:subplot_index], bbox(0,0,1,1)))
#     twinsp = sp.plt.subplots[end]
#     twinsp[:xaxis][:mirror] = true
#     twinsp[:background_color_inside] = RGBA{Float64}(0,0,0,0)
#     Plots.link_axes!(sp[:yaxis], twinsp[:yaxis])
#     return twinsp
# end
# twiny(plt::Plots.Plot = current()) = twiny(plt[1])
# #
@everywhere function number_decimals(x::Number)::Int
    ndigits = 0
    while !isinteger(x) && ndigits < 4 # gardefou pour les nombres à écriture décimale infinie
        x = x * 10
        ndigits += 1
    end

    return ndigits
end

@everywhere function grad2(thetas) # returns ||∇θ(x,t)||²
    L     = size(thetas)[1]
    grad2 = zeros(L,L)
    # thetas= mod.(thetas,2π)
    for j in 1:L
        for i in 1:L
            neighbours = get_angle_nearest_neighbours(thetas,i,j,L)
            dθdx       = (neighbours[1] - neighbours[2]) / 2
            dθdy       = (neighbours[3] - neighbours[4]) / 2
            grad2[i,j] = max(abs(dθdx), abs(dθdy))
            # grad2[i,j] = 1/min(abs(dθdx),abs(dθdy))
            # grad2[i,j] = dθdx^2 + dθdy^2
        end
    end
    return grad2
end

@everywhere function walls(thetas;tol=π/4)
    L   = size(thetas)[1]
    walls = zeros(L,L)
    # tol_aligned = π/4
    # tol_perp    =
    for j in 1:L
        for i in 1:L
            bool_vertical   = abs(thetas[mod1(i+1,L),j] - thetas[mod1(i-1,L),j]) < tol && abs(π - abs(thetas[i,mod1(j+2,L)] - thetas[i,mod1(j+2,L)])) < tol
            bool_horizontal = abs(π - abs(thetas[mod1(i+2,L),j] - thetas[mod1(i-2,L),j])) < tol && abs(thetas[i,mod1(j+1,L)] - thetas[i,mod1(j+1,L)]) < tol
            # bool_diagonal   = abs(π - abs(thetas[mod1(i+1,L),j] - thetas[mod1(i-1,L),j])) < tol && abs(thetas[i,mod1(j+1,L)] - thetas[i,mod1(j+1,L)]) < tol
            walls[i,j] = bool_horizontal || bool_vertical
        end
    end
    return walls
end

@everywhere function energy(thetas)
    L     = size(thetas)[1]
    energy_local = zeros(L,L)
    for j in 1:L
        for i in 1:L
            neighbours = get_angle_nearest_neighbours(thetas,i,j,L)
            θ = thetas[i,j]
            energy_local[i,j] = -sum(cos.(neighbours .- θ))
        end
    end
    energy_global = sum(energy_local)
    return energy_local,energy_global
end

@everywhere function psi(thetas)
    L     = size(thetas)[1]
    m = [mean(cos.(thetas)),mean(sin.(thetas))]
    u = m / (sqrt(m[1]^2 + m[2]^2))

    psi = zeros(size(thetas))
    for i in eachindex(thetas)
        psi[i] = u[1]*cos(thetas[i]) + u[2]*sin(thetas[i])
    end
    return psi
end

@everywhere function grad_sq(matrix)
    L     = size(thetas)[1]
    grad_sq = zeros(L-2,L-2)
    for i in 2:L-1
        for j in 2:L-1
            dx = (matrix[i+1] - matrix[i-1])/2
            dy = (matrix[j+1] - matrix[j-1])/2
            grad_sq[i-1,j-1] = sqrt(dx^2 + dy^2)
        end
    end
    return grad_sq
end
