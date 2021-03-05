
"Copyright (c) 2020 Y.Rouzaire All Rights Reserved."

## Handling locations as Tuple{Int16,Int16}

function dist(a::Union{Tuple{Int16,Int16},Missing},b::Union{Tuple{Int16,Int16},Missing},L::Int)::Union{Float64,Missing}  # euclidian distance with Periodic BCs
    if ismissing(a) || ismissing(b) return missing
    else
        dx = abs(a[1] - b[1]) ; dx = min(dx,L-dx)
        dy = abs(a[2] - b[2]) ; dy = min(dy,L-dy)
        return sqrt(dx^2 + dy^2)
    end
end

function dist_missing(a::Tuple{Union{Missing,Int16},Union{Missing,Int16}},b::Tuple{Union{Missing,Int16},Union{Missing,Int16}},L::Int)::Union{Float64,Missing}  # euclidian distance with Periodic BCs
    if a === (missing,missing) || b === (missing,missing) return missing
    else
        dx = abs(a[1] - b[1]) ; dx = min(dx,L-dx)
        dy = abs(a[2] - b[2]) ; dy = min(dy,L-dy)
        return sqrt(dx^2 + dy^2)
    end
end

function dist(a::Tuple{Number,Number},b::Tuple{Number,Number},L::Int)  # euclidian distance with Periodic BCs
    dx = abs(a[1] - b[1]) ; dx = min(dx,L-dx)
    dy = abs(a[2] - b[2]) ; dy = min(dy,L-dy)
    return sqrt(dx^2 + dy^2)
end

function norm(v::Tuple{Int16,Int16})::Float64
    return sqrt(v[1]^2 + v[2]^2)
end


function inner_product(v1::Tuple{Int16,Int16},v2::Tuple{Int16,Int16})::Float64
    return v1[1]*v2[1] + v1[2]*v2[2]
end

function cos_angle_vectors(vector_before::Tuple{Int16,Int16},vector_after::Tuple{Int16,Int16})::Float64
    if    vector_before == (0,0) error("No mouvement ")
    elseif vector_after == (0,0) error("No mouvement")
    else return (inner_product(vector_before,vector_after)/norm(vector_before)/norm(vector_after)) end
end

## Creation of lattices
function create_isolated_vortex(H,W,q)
    @assert abs(q) == 1
    thetas = zeros(H,W)
    H2 = round(Int,H/2)
    W2 = round(Int,W/2)
    for j in 1:W
        for i in 1:H
            y = H2 - i
            x = j - W2
            thetas[i,j] = q * atan(y,x)
        end
    end
    return thetas
end

function create_pair_vortices(L,r0)
    #= Check for meaningfulness of the defaults separation,
    otherwise the defaults will annihilate with relaxation =#
    @assert r0 ≤ 0.75L "Error : r0 is too large. "
    @assert iseven(r0) "Error : r0 has to be even. "

    L2  = round(Int,L/2)
    r02 = floor(Int,r0/2)

    #= Create each default separately and juxtapose them.
    Very interesting fact : the anti-default can be created
    by flipping one dimension of the defaut (hence the reverse
    operation performed by "end:-1:1") =#
    default  = create_isolated_vortex(L,L2,1)
    thetas   = cat(default[:,end:-1:1],default,dims=2)

    # Smooth domain walls in order not to create news vortices when relaxing
    thetas[1:3,:] = thetas[end-2:end,:] = zeros(3,L)

    # Enforce the distance between defaults
    if r0 < L2
        extra = L2 - r0
        thetas[:,extra+1:L2+floor(Int,extra/2)] = thetas[:,1:L2-floor(Int,extra/2)]
        for i in 1:extra
            thetas[:,i] = thetas[:,end]
        end
    elseif r0 > L2
        lack = r0 - L2
        thetas[:,1:L2-lack] = thetas[:,lack:L2-1]
        for i in L2-lack:L2
            thetas[:,i] = thetas[:,L2+1]
        end
    end

    # Let the system relax, while enforcing periodic BCs
    thetas = relax(thetas)

    return thetas
end

function relax(thetas) # Let the system relax at T = Var = 0
    L = size(thetas)[1]

    t = 0.0 ; dt = determine_dt(0,0) ; t_relax = 2 # t_relax = 2 is enough to smoothen the manually imposed PBCs
    while t<t_relax
        thetas_new = thetas
        for j in 1:L
            for i in 1:L
                angle_neighbours = get_angle_nearest_neighbours(thetas,i,j,L) # O(1)
                θ = thetas[i,j]
                sin_angle_neighbours = sin.(angle_neighbours .- θ)
                thetas_new[i,j] =  θ + dt*sum(sin_angle_neighbours)
            end
        end
        t += dt
        thetas = thetas_new
    end
    return thetas
end

function determine_dt(T,Var)
    L = 200
    first_try = maximum(abs.(rand(Normal(0,sqrt(Var)),L,L)))
    for m in 1:25
        proposal = maximum(abs.(rand(Normal(0,sqrt(Var)),L,L)))
        if proposal > first_try first_try = proposal end
    end
    arbitrary_coeff =  π/10
    dt = min(arbitrary_coeff/first_try , arbitrary_coeff/2 , arbitrary_coeff^2*π/4/T)
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
            it is sqrt(dt) and not dt as is that enters the Euler Maryuama integrator.
            It is the more constraining term for usual ranges of \sigma and T.
        5. The overall coefficient π/10 is arbitrary and can be modulated as desired.
    =#
        return dt
end

function determine_dt(TV)
    Tmax = 0
    Varmax = 0
    for element in TV
        Tmax   = max(element[1],Tmax)
        Varmax = max(element[2],Varmax)
    end
    return determine_dt(Tmax,Varmax)
end
## Get info about nearest spins
function get_angle_nearest_neighbours(thetas::Matrix{Float64},i::Int,j::Int,L::Int)::Vector{Float64}
    return [thetas[mod1(i+1,L),j], ## Note : mod1 instead of mod to deal with the indexing at from 1
            thetas[mod1(i-1,L),j],
            thetas[i,mod1(j+1,L)],
            thetas[i,mod1(j-1,L)] ]
end

function get_angle_nearest_neighbours_FBC(thetas::Matrix{Float64},i::Int,j::Int,L::Int)::Vector{Float64}
    neighbours = Float64[]
    if i>1 push!(neighbours,thetas[i-1,j]) end
    if i<L push!(neighbours,thetas[i+1,j]) end
    if j>1 push!(neighbours,thetas[i,j-1]) end
    if j<L push!(neighbours,thetas[i,j+1]) end

    return neighbours
end

function get_omega_nearest_neighbours(omegas::Matrix{Float64},i::Int,j::Int,L::Int)::Vector{Float64}
    return [omegas[mod1(i+1,L),j], ## Note : mod1 instead of mod to deal with the indexing at from 1
            omegas[mod1(i-1,L),j],
            omegas[i,mod1(j+1,L)],
            omegas[i,mod1(j-1,L)] ]
end

## Gestion des Vortex
function arclength(theta1::Float64,theta2::Float64)::Float64
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

function get_vorticity(thetas::Matrix{Float64},i::Int,j::Int,L::Int)::Int
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

function spot_pair_defaults_global(thetas::Matrix{Float64})
    L = size(thetas)[1]
    list_defaults = Tuple{Int16,Int16,Int16}[]
    for i in 1:L
        for j in 1:L
            q = get_vorticity(thetas,i,j,L)
            if q ≠ 0
                push!(list_defaults,(i,j,q))
            end
        end
    end
    # @assert length(list_defaults) ≤ 2 # has to remain
    if !isempty(list_defaults)
        list_defaults_convention = Vector{Tuple{Int16,Int16}}(undef,2)
        for i in eachindex(list_defaults)
            list_defaults_convention[2-Int((list_defaults[i][3]+1)/2)] = list_defaults[i][1:2]
            #= Note : Int(2-(q+1)/2), where q=list_defaults[i][3] ensures that
            the vortex comes first, and the antivortex comes next. =#
        end
        return list_defaults_convention
    else error("No defaults.")
    end
    return Tuple{Int16,Int16}[] # it means there is no vortices
    # changer ce vieux mode de return pas consistent desfois des tuples, desfois des matrices
end

function spot_pair_defaults_local(thetas::Matrix{Float64},V::Tuple{Int16,Int16},Vbar::Tuple{Int16,Int16})::Tuple{Vector{Tuple{Int16,Int16}},Bool}
    # V    is the last location (x,y) of the vortex. By convention, q=+1 for the vortex.
    # Vbar is the last location (x,y) of the antivortex. By convention, q=-1 for the antivortex.
    # In spot_single_default_local(), the first location if the one of the default of interest, the second is the other of the pair.
    new_location_vortex,alone_v        = spot_single_default_local(thetas,V,Vbar,+1)
    new_location_antivortex,alone_vbar = spot_single_default_local(thetas,Vbar,V,-1)
    return [new_location_vortex,new_location_antivortex],(alone_v && alone_vbar)
    #= Note that an error thrown in the first call to spot_single_default_local()
    will stop the function. That's good because if we lost a vortex (out of lattice
    or uncontrolled annihilation), the simulation should stop. If the error is due
    to a genuine annihilation, then the second call would have raised the same error.=#
end

function spot_pair_defaults_local_PBC(thetas::Matrix{Float64},V::Tuple{Int16,Int16},Vbar::Tuple{Int16,Int16})::Tuple{Vector{Tuple{Int16,Int16}},Bool}
    # V    is the last location (x,y) of the vortex. By convention, q=+1 for the vortex.
    # Vbar is the last location (x,y) of the antivortex. By convention, q=-1 for the antivortex.
    # In spot_single_default_local(), the first location if the one of the default of interest, the second is the other of the pair.
    new_location_vortex,alone_v        = spot_single_default_local_PBC(thetas,V,Vbar,+1)
    new_location_antivortex,alone_vbar = spot_single_default_local_PBC(thetas,Vbar,V,-1)
    return [new_location_vortex,new_location_antivortex],(alone_v && alone_vbar)
    #= Note that an error thrown in the first call to spot_single_default_local()
    will stop the function. That's good because if we lost a vortex (out of lattice
    or uncontrolled annihilation), the simulation should stop. If the error is due
    to a genuine annihilation, then the second call would have raised the same error.=#
end


function spot_single_default_global(thetas::Matrix{Float64})::Tuple{Tuple{Int16,Int16},Int}
    L = size(thetas)[1]
    list_defaults = Tuple{Int16,Int16,Int}[]
    for i in 2:L-1
        for j in 2:L-1
            q = get_vorticity(thetas,i,j,L)
            if q ≠ 0
                push!(list_defaults,(i,j,q))
            end
        end
    end
    # @assert length(list_defaults) == 1 "Il y a $(length(list_defaults)) vortex"# has to remain
    # println("Defaults : $list_defaults")
    return list_defaults[1][1:2],list_defaults[1][3]
end

function spot_single_default_local(thetas::Matrix{Float64},last_loc::Tuple{Int16,Int16},known_loc::Tuple{Int16,Int16},Q::Int,margin::Int=6)::Tuple{Tuple{Int16,Int16},Bool}
    # V    is the location (x,y,q) of the default
    #= The search for the new location of the default will scan a square of
    dimension 2margin x 2margin (within the limits of the LxL lattice) ,
    with the last location as center. The whole method is O(margin²). =#
    L = size(thetas)[1]
    positions = []

    # Dimensions of the square
    j_low,j_high = max(1,last_loc[2]-margin) , min(L,last_loc[2]+margin)
    i_low,i_high = max(1,last_loc[1]-margin) , min(L,last_loc[1]+margin)
    for j in j_low:j_high
        for i in i_low:i_high
            q = get_vorticity(thetas,i,j,L)
            if q ≠ 0
                push!(positions,(i,j,q))
            end
        end
    end


    ℓ = length(positions)
    #= Summary of what follows :
    ℓ = 0 : (i)   Controled Annihilation (encounter of the vortex with its antivortex)
            (ii)  Unexpected Annihilation (encounter of the vortex with another antivortex
            (iii) The (single) vortex left the lattice.
            In all three case, throw an explicit error and treat it later on.

    ℓ = 1 : All good ! One is pretty sure that the only detected vortex is the same
    than the one of the previous timestep. The only case where we could be mistaken
    is if a pair of vortex enters the square and that our previous vortex annihilates
    with the newcomer antivortex. In this case, the only default remaining would be
    the newcomer vortex, and we would be clueless about it. The only signature would be a
    possible jump in space in the r(t) graph.

    ℓ = 2 : If the other default is known/authorized, meaning that it is the former
    antivortex of the default we currently work on, that's not important and we keep
    going as if the vortex was alone. If the other defaut is NOT known/authorized,
    it will perturb the displacment of our vortex : don't save the current position
    by signalling it (alone = false)

    ℓ ≥ 3 : We are sure to be in the bad case #2.  =#
    if ℓ == 1
        alone = true # no other default in the searched region
        most_probable = positions[1][1:2]
        #= Actually,  =#
    elseif ℓ > 1
        alone = false # by default. We deal we the special case ℓ = 2 just below.

        if ℓ == 2 && ((known_loc[1],known_loc[2],-Q) in positions) ; alone = true ; end
        #= If ℓ == 2, one has 2 possibilities :
            - either the extra default is "known" (its the antidefault of
            the one we currently consider), in which case turn alone = true.
            - or it's not, in which case we leave alone = false.
            In any case, the following routine to determine the location of
            the default currenlty tracked is still completely valid.
        Note that when using this function for the study of a single vortex,
        one needs to provide an impossible known_loc, such as (-1,-1). =#

        distances_to_last = [dist(positions[i][1:2],last_loc,L) for i in 1:ℓ]
        # Kill candidates with opposite polarity
        for i in 1:ℓ
            element = positions[i]
            if element[3] ≠ Q distances_to_last[i] = Inf end
        end
        most_probable = positions[sortperm(distances_to_last)][1]
        #= Returns the position with the smallest distance to the last location,
        hence we choose that one as the most probable candidate for the vortex we
        are considering. =#

    else # dealing with the case ℓ = 0

        close_from_boundary = last_loc[1] < 4 || last_loc[1] > L - 4 || last_loc[2] < 4 || last_loc[2] > L - 4
        if close_from_boundary                      error("The vortex left the lattice.")
        elseif dist(last_loc,known_loc,L) ≤ sqrt(5) error("Controlled annihilation.") # sqrt(5) is arbitrary
        else                                        error("Unexpected annihilation.") end
    end
    return most_probable[1:2],alone
end

function spot_single_default_local_PBC(thetas::Matrix{Float64},last_loc::Tuple{Int16,Int16},known_loc::Tuple{Int16,Int16},Q::Int,margin::Int=6)::Tuple{Tuple{Int16,Int16},Bool}
    # V    is the location (x,y,q) of the default
    #= The search for the new location of the default will scan a square of
    dimension 2margin x 2margin (within the limits of the LxL lattice) ,
    with the last location as center. The whole method is O(margin²). =#
    L = size(thetas)[1]
    positions = []

    # Dimensions of the square
    i_low,i_high = last_loc[1]-margin , last_loc[1]+margin
    j_low,j_high = last_loc[2]-margin , last_loc[2]+margin
    for j in j_low:j_high
        for i in i_low:i_high
            q = get_vorticity(thetas,mod1(i,L),mod1(j,L),L)
            if q ≠ 0
                push!(positions,(mod1(i,L),mod1(j,L),q))
            end
        end
    end


    ℓ = length(positions)
    if ℓ == 1
        alone = true # no other default in the searched region
        most_probable = positions[1][1:2]
    elseif ℓ > 1
        alone = false # by default. We deal we the special case ℓ = 2 just below.

        if ℓ == 2 && ((known_loc[1],known_loc[2],-Q) in positions) ; alone = true ; end

        distances_to_last = [dist(positions[i][1:2],last_loc,L) for i in 1:ℓ]
        # Kill candidates with opposite polarity
        for i in 1:ℓ
            element = positions[i]
            if element[3] ≠ Q distances_to_last[i] = Inf end
        end
        most_probable = positions[sortperm(distances_to_last)][1]

    else # dealing with the case ℓ = 0
        close_from_boundary = last_loc[1] < 4 || last_loc[1] > L - 4 || last_loc[2] < 4 || last_loc[2] > L - 4
        if dist(last_loc,known_loc,L) ≤ 2 error("Controlled annihilation.") # value is arbitrary
        elseif close_from_boundary        error("The vortex left the lattice.")
        else                              error("Unexpected annihilation.") end
    end
    return most_probable[1:2],alone
end


# function spot_single_default_local(thetas::Matrix{Float64},last_loc::Tuple{Int16,Int16},Q::Int,margin::Int=10)::Tuple{Tuple{Int16,Int16},Bool}
#     # V    is the location (x,y,q) of the default
#     #= The search for the new location of the default will scan a square of
#     dimension 2margin x 2margin (within the limits of the LxL lattice) ,
#     with the last location as center. The whole method is O(margin²). =#
#     L = size(thetas)[1]
#     positions = []
#
#     # Dimensions of the square
#     j_low,j_high = max(1,last_loc[2]-margin) , min(L,last_loc[2]+margin)
#     i_low,i_high = max(1,last_loc[1]-margin) , min(L,last_loc[1]+margin)
#     for j in j_low:j_high
#         for i in i_low:i_high
#             q = get_vorticity(thetas,i,j,L)
#             if q ≠ 0
#                 push!(positions,(i,j,q))
#             end
#         end
#     end
#
#     ℓ = length(positions)
#     if ℓ == 1
#         alone = true # no other default in the searched region
#         most_probable = positions[1][1:2]
#         #= Actually, one is pretty sure that the only detected vortex is the same
#         than the one of the previous timestep. The only case where we could be mistaken
#         is if a pair of vortex enters the square, that our previous vortex annihilates
#         with the newcomer antivortex. In this case, the only default remaining would be
#         the newcomer vortex, and we are clueless about it. The only signature would be a
#         possible jump in space in the r(t) graph. =#
#     elseif ℓ > 1
#         alone = false
#         distances_to_last = [dist(positions[i][1:2],last_loc,L) for i in 1:ℓ]
#         # Kill candidates with opposite polarity
#         for i in 1:ℓ
#             element = positions[i]
#             if element[3] ≠ Q distances_to_last[i] = Inf end
#         end
#         most_probable = positions[sortperm(distances_to_last)][1]
#         #= Returns the position with the smallest distance to the last location,
#         hence we choose that one as the most probable candidate for the vortex we
#         are considering. =#
#     else
#         close_from_boundary = last_loc[1] < 4 || last_loc[1] > L - 4 || last_loc[2] < 4 || last_loc[2] > L - 4
#         if close_from_boundary error("Error : ℓ = $ℓ. Most probable cause : the vortex left the lattice.")
#         else error("Error : ℓ = $ℓ. Most probable cause : the vortex got annihilated by an errant antivortex.") end
#     end
#     return most_probable[1:2],alone
# end

# function spot_single_default_local(thetas::Matrix{Float64},last_loc::Tuple{Int16,Int16,Int},margin::Int=10)::Tuple{Tuple{Int16,Int16,Int},Bool}
#     # V    is the location (x,y,q) of the default
#     #= The search for the new location of the default will scan a square of
#     dimension 2margin x 2margin (within the limits of the LxL lattice) ,
#     with the last location as center. The whole method is O(margin²). =#
#     L = size(thetas)[1]
#     positions = []
#
#
#     # Dimensions of the square
#     j_low,j_high = max(1,last_loc[2]-margin+1) , min(L,last_loc[2]+margin-1)
#     i_low,i_high = max(1,last_loc[1]-margin+1) , min(L,last_loc[1]+margin-1)
#     for j in j_low:j_high
#         for i in i_low:i_high
#             q = get_vorticity(thetas,i,j,L)
#             if q ≠ 0
#                 push!(positions,(i,j,q))
#             end
#         end
#     end
#
#     ℓ = length(positions)
#     if ℓ == 1
#         alone = true # no other default in the searched region
#         most_probable = positions[1][1:2]
#         #= Actually, one is pretty sure that the only detected vortex is the same
#         than the one of the previous timestep. The only case where we could be mistaken
#         is if a pair of vortex enters the square, that our previous vortex annihilates
#         with the newcomer antivortex. In this case, the only default remaining would be
#         the newcomer vortex, and we are clueless about it. The only signature would be a
#         possible jump in space in the r(t) graph. =#
#     elseif ℓ > 1
#         alone = false
#         distances_to_last = [dist(positions[i][1:2],last_loc,L) for i in 1:ℓ]
#         most_probable = positions[sortperm(distances_to_last)][1]
#         #= Returns the position with the smallest distance to the last location,
#         hence we choose that one as the most probable candidate for the vortex we
#         were considering. =#
#     else
#         close_from_boundary = last_loc[1] < 4 || last_loc[1] > L - 4 || last_loc[2] < 4 || last_loc[2] > L - 4
#         if close_from_boundary error("Error : ℓ = $ℓ. Most probable cause : the vortex left the lattice.")
#         else error("Error : ℓ = $ℓ. Most probable cause : the vortex got annihilated by an errant antivortex.") end
#     end
#     return most_probable,alone
# end



## Time evolution Periodic BCs
function update(thetas::Matrix{Float64},omegas::Matrix{Float64},L::Int,T::Number,dt::Float64)::Matrix{Float64}
    noise = sqrt(2*T*dt)*randn(L,L)
    thetas_new = Array{Float64,2}(undef,L,L)
    for j in 1:L
        for i in 1:L
            #= NB : the inversion j,i is wanted. Because Julia accesses the matrices by columns and not by rows,
            for i,j leads to O(L^3) while for j,i leads to O(L^2), as expected.  =#
            angle_neighbours = get_angle_nearest_neighbours(thetas,i,j,L) # O(1)
            θ,ω = thetas[i,j] , omegas[i,j]
            sin_angle_neighbours = sin.(angle_neighbours .- θ)
            thetas_new[i,j] =  θ + ω*dt + dt*sum(sin_angle_neighbours) + noise[i,j] # O(1)
        end
    end
    return thetas_new
end

# function evolve(thetas::Matrix{Float64},omegas::Matrix{Float64},T::Number,dt::Float64,tmax::Number,save_every=1)
#     L = size(thetas)[1]
#     nsteps_max = round(Int,tmax/dt)
#     history_locations = Vector{Union{Tuple{Int16,Int16},Missing}}(missing,2,1+floor(Int,nsteps_max/save_every))
#     # Initial location
#     last_location = spot_single_default_global(thetas)
#     history_locations[1] = last_location
#
#     nsteps_max = round(Int,tmax/dt)
#     # Time Evolution
#     for i in 1:nsteps_max
#         thetas = update(thetas,omegas,L,T,dt) # O(L²)
#         if i%100 == 0 # every 100 time steps, scan the whole lattice to check that there is no default creation
#             push!(vortex_location,spot_defaults_global(thetas)) # O(L²)
#         elseif i%save_every == 0
#             push!(vortex_location,spot_defaults_local(thetas,vortex_location[end][1],vortex_location[end][2])) # O(1)
#         end
#     end
#
#     return hcat(vortex_location...) # returns a 2 x nsave Matrix
# end
#
# function evolve_Pair(thetas::Matrix{Float64},omegas::Matrix{Float64},T::Number,dt::Float64,tmax::Number,save_every::Int)
#     L = size(thetas)[1]
#     nsteps_max = floor(Int,tmax/dt)
#     history_locations = Matrix{Union{Tuple{Int16,Int16},Missing}}(missing,2,1+floor(Int16,nsteps_max/save_every))
#     history_distance  = Vector{Union{Float64,Missing}}(missing,1+floor(Int16,nsteps_max/save_every))
#     # Initial values
#     last_location = spot_pair_defaults_global(thetas) # type = Vector{Tuple{Int16,Int16}}(undef,2)
#     history_locations[:,1] = last_location
#     history_distance[1] = dist(last_location[1],last_location[2],L) # returns Union{Float64,Missing}
#
#     # Time Evolution
#     for i in 1:nsteps_max
#         thetas = update(thetas,omegas,L,T,dt) # O(L²)
#         if i%save_every == 0
#             ii = 1+div(i,save_every) # index to access the arrays to store information
#             try
#                 last_location,alone = spot_pair_defaults_local(thetas,last_location[1],last_location[2])
#                 if alone # there are no unknown vortices in the surroundings (the '2margin' square)
#                     saved_location = last_location
#                     saved_distance = dist(last_location[1],last_location[2],L)
#                 else # there at least one unknown vortex in the surroundings that could pollute the data
#                     # saved_location = [missing,missing] # to be cautious
#                     saved_location = last_location # to be sure to have data, even if it might be polluted
#                     saved_distance = missing
#                 end
#                 history_locations[:,ii] = saved_location
#                 history_distance[ii]    = saved_distance
#             catch e # we lost at least one vortex (out of lattice, annihilation under control, unpredicted annihilation)
#                 println(e)
#                 if e == ErrorException("Controlled annihilation.")
#                     history_distance[ii:end] .= 0.0 # since the pair has annihilated, the distance is 0 for the rest of the simulation
#                 end
#                 break # we leave all the other remaining values as they are (missing) and return histories
#             end
#         end
#
#     end
#
#     return history_locations,history_distance
# end


function evolve_Pair(thetas::Matrix{Float64},omegas::Matrix{Float64},T::Number,dt::Float64,tmax::Number,save_every::Int)
    L = size(thetas)[1]
    nsteps_max = floor(Int,tmax/dt)
    history_locations = Matrix{Tuple{Union{Int16,Missing},Union{Int16,Missing}}}(undef,2,1+floor(Int16,nsteps_max/save_every))
    for i in eachindex(history_locations) history_locations[i] = (missing,missing) end
    history_distance  = Vector{Union{Float64,Missing}}(missing,1+floor(Int16,nsteps_max/save_every))

    # Initial values
    last_location = spot_pair_defaults_global(thetas) # type = Vector{Tuple{Int16,Int16}}(undef,2)
    history_locations[:,1] = last_location
    history_distance[1] = dist(last_location[1],last_location[2],L) # returns Union{Float64,Missing}

    # Time Evolution
    for i in 1:nsteps_max
        thetas = update(thetas,omegas,L,T,dt) # O(L²)
        if i%save_every == 0
            ii = 1+div(i,save_every) # index to access the arrays to store information
            try
                last_location,alone = spot_pair_defaults_local_PBC(thetas,last_location[1],last_location[2])
                if alone # there are no unknown vortices in the surroundings (the '2margin' square)
                    saved_location = last_location
                    saved_distance = dist(last_location[1],last_location[2],L)
                else # there at least one unknown vortex in the surroundings that could pollute the data
                    #To be cautious
                        # saved_location = [(missing,missing),(missing,missing)] # to be cautious
                        # saved_distance = missing
                    # To be sure to have data, even if it might be polluted
                        saved_location = last_location
                        saved_distance = dist(last_location[1],last_location[2],L)

                end
                history_locations[:,ii] = saved_location
                history_distance[ii]    = saved_distance
            catch e # we lost at least one vortex (out of lattice, annihilation under control, unpredicted annihilation)
                println(e)
                if e == ErrorException("Controlled annihilation.")
                    history_distance[ii:end] .= 0.0 # since the pair has annihilated, the distance is 0 for the rest of the simulation
                end
                break # we leave all the other remaining values as they are (missing) and return histories
            end
        end

    end

    return history_locations,history_distance
end
## Time evolution Free BCs
function update_FBC(thetas::Matrix{Float64},omegas::Matrix{Float64},L::Int,T::Number,dt::Float64)::Matrix{Float64}
    noise = sqrt(2*T*dt)*randn(L,L)
    thetas_new = Array{Float64,2}(undef,L,L)
    for j in 1:L
        for i in 1:L
            #= NB : the inversion j,i is wanted. Because Julia accesses the matrices by columns and not by rows,
            for i,j leads to O(L^3) while for j,i leads to O(L^2), as expected.  =#
            angle_neighbours = get_angle_nearest_neighbours_FBC(thetas,i,j,L) # O(1)
            θ,ω = thetas[i,j] , omegas[i,j]
            sin_angle_neighbours = sin.(angle_neighbours .- θ)
            thetas_new[i,j] =  θ + ω*dt + dt*sum(sin_angle_neighbours) + noise[i,j] # O(1)
        end
    end
    return thetas_new
end

function evolve_FBC(Q::Int,thetas::Matrix{Float64},omegas::Matrix{Float64},T::Number,dt::Float64,tmax::Number,save_every::Int)
    L = size(thetas)[1]
    nsteps_max = floor(Int,tmax/dt)
    history_locations = Vector{Union{Tuple{Int16,Int16},Missing}}(missing,1+floor(Int,nsteps_max/save_every))
    # Initial location
    last_location,~ = spot_single_default_global(thetas)
    history_locations[1] = last_location

    # Time Evolution
    for i in 1:nsteps_max
        thetas = update_FBC(thetas,omegas,L,T,dt) # O(L²)
        if i%save_every == 0
            try
                lastknown = (Int16(-1),Int16(-1))
                last_location,alone = spot_single_default_local(thetas,last_location,lastknown,Q)
                if alone # there are no other vortices in the surroundings (the '2margin' square)
                    saved_location = last_location
                else # there are other vortices in the surroundings that could pollute the data
                    saved_location = missing
                end
                history_locations[1+div(i,save_every)] = saved_location  # O(1)
            catch e  # we lost our vortex (out of lattice, annihilation)
                println(e)
                printstyled("Warning : Vortex lost, simulation stopped at t = $(dt*i). \n"; color = :yellow)
                break # we leave all the remaining values as they are : missing
            end
        end

    end

    return history_locations
end

## Data Analysis
# Numerics
function remove_negative(series,threshold=0)
    # result = Vector{Union{Float64,Missing}}(undef,length(series)) # initializes by default to "missing"
    result = Vector{Number}(undef,length(series)) # initializes by default to "missing"
    for i in eachindex(series)
        if series[i] > threshold
            result[i] = series[i]
        else result[i] = NaN
        end
    end
    return result
end
# Physics
function SD(L,nT,nVar,R,locations::Array{Union{Missing, Tuple{Int16,Int16}},4})
    SD = Array{Union{Number,Missing}}(undef,(length(0:save_every:tmax/dt),nT,nVar,R))
    for r in 1:R
        for j in 1:nVar
            for i in 1:nT
                for t in 1:size(SD)[1]
                    SD[t,i,j,r] = dist(locations[t,i,j,r], locations[1,i,j,r],L)^2
                end
            end
        end
    end
    MSD_avg = Array{Number}(undef,(size(SD)[1],nT,nVar))
    MSD_std = Array{Number}(undef,(size(SD)[1],nT,nVar))
    for j in 1:nVar
        for i in 1:nT
            for t in 1:size(SD)[1]
                MSD_avg[t,i,j] = mean(skipmissing(SD[t,i,j,:]))
                MSD_std[t,i,j] =  std(skipmissing(SD[t,i,j,:]))
            end
        end
    end

    return MSD_avg,MSD_std
end

function SD(L,nT,nVar,R,locations::Array{Union{Missing, Tuple{Int16,Int16}},5})
    SD = Array{Union{Number,Missing}}(undef,(length(0:save_every:tmax/dt),nT,nVar,R))
    for r in 1:R
        for j in 1:length(Vars)
            for i in 1:length(Ts)
                for t in 1:size(SD)[1]
                    rt = dist(locations[1,t,i,j,r],locations[2,t,i,j,r],L)
                    r0 = dist(locations[1,1,i,j,r],locations[2,1,i,j,r],L)
                    SD[t,i,j,r] = (rt - r0)^2
                end
            end
        end
    end
    MSD_avg = Array{Number}(undef,(size(SD)[1],nT,nVar))
    MSD_std = Array{Number}(undef,(size(SD)[1],nT,nVar))
    for j in 1:length(Vars)
        for i in 1:length(Ts)
            for t in 1:size(SD)[1]
                MSD_avg[t,i,j] = mean(skipmissing(SD[t,i,j,:]))
                MSD_std[t,i,j] =  std(skipmissing(SD[t,i,j,:]))
            end
        end
    end

    return MSD_avg,MSD_std
end

# function remove_duplicates(series)::Vector{Tuple{Int16,Int16}}
function remove_duplicates(series::Array{Union{Missing,Tuple{Int16,Int16}}})::Vector{Tuple{Int16,Int16}}
    continuous_series = Tuple{Int16,Int16}[]
    push!(continuous_series,series[1])
    last = series[1]
    for i in 2:length(series)
        if  !ismissing(series[i]) && !ismissing(series[i-1]) && !ismissing(series[i][1]) && !ismissing(series[i-1][1]) && (series[i] ≠ last)
            push!(continuous_series,series[i])
            last = series[i]
        end
    end
    return continuous_series
end

function average_cos(series::Vector{Tuple{Int16,Int16}})
    tmp = Float64[]
    for i in 2:length(series)-1
        vecteur_before = series[i]   .- series[i-1]
        vecteur_after  = series[i+1] .- series[i]
        push!(tmp,cos_angle_vectors(vecteur_before,vecteur_after))
    end
    return mean(tmp)
    # return mean(tmp)
end

function stiffnessa(locations::Array{Union{Missing,Tuple{Int16,Int16}},5})
    #= Returns ⟨⟨cos(ϕ_i)⟩⟩
    where ϕ_i is the angle between the vector r_i and r_(i+1),
    where the innermost average is on i (the time),
    and the outermost average is on realisations. =#
    sp,nt,nT,nVar,R = size(locations)
    cosϕ = Array{Union{Number,Missing}}(undef,(sp,nT,nVar,R))
    for r in 1:R
        for j in 1:nVar
            for i in 1:nT
                for v in 1:sp
                    continuous_chain = remove_duplicates(locations[v,:,i,j,r])
                    cosϕ[v,i,j,r]    = average_cos(continuous_chain)
                end
            end
        end
    end

    # Average over realisations
    stiffness_avg_real = Array{Number}(undef,(sp,nT,nVar))
    for j in 1:nVar
        for i in 1:nT
            for v in 1:sp
                stiffness_avg_real[v,i,j] = mean(cosϕ[v,i,j,:])
            end
        end
    end

    return stiffness_avg_real
end


function stiffness(locations::Array{Union{Missing,Tuple{Int16,Int16}},4})
    #= Returns ⟨⟨cos(ϕ_i)⟩⟩
    where ϕ_i is the angle between the vector r_i and r_(i+1),
    where the innermost average is on i (the time),
    and the outermost average is on realisations. =#
    nt,nT,nVar,R = size(locations)
    cosϕ = Array{Union{Number,Missing}}(undef,(nT,nVar,R))
    for r in 1:R
        for j in 1:nVar
            for i in 1:nT
                continuous_chain = remove_duplicates(locations[:,i,j,r])
                for i in 2:length(continuous_chain)
                    @assert continuous_chain[i] ≠ continuous_chain[i-1] println("Insight : ",continuous_chain[i-2:i+2])
                end
                cosϕ[i,j,r]      = average_cos(continuous_chain)
            end
        end
    end

    # Average over realisations
    stiffness_avg_real = Array{Number}(undef,(nT,nVar))
    for j in 1:nVar
        for i in 1:nT
            stiffness_avg_real[i,j] = mean(cosϕ[i,j,:])
        end
    end

    return stiffness_avg_real
end

## For GIFs

function trajectory(L,T,Var,tmax,dt,Q,save_every,R)
    trajs   = Matrix{Union{Missing,Tuple{Int,Int}}}(undef,1+floor(Int,floor(Int,tmax/dt)/save_every),R)
    Threads.@threads for r in 1:R
        thetas = create_isolated_vortex(L,L,+1)
        omegas = rand(Normal(0,sqrt(Var)),L,L)
        trajs[:,r] = evolve_FBC(Q,thetas,omegas,T,dt,tmax,save_every)
    end

    trajs_aux = Matrix{Tuple{Union{Missing,Int},Union{Missing,Int}}}(undef,1+floor(Int,floor(Int,tmax/dt)/save_every),R)
    for i in eachindex(trajs)
        if ismissing(trajs[i]) trajs_aux[i] = (missing,missing)
        else trajs_aux[i] = Int64.(trajs[i])
        end
    end

    return trajs_aux
end

function trajectory_and_angles(L,T,Var,tmax,dt,Q,save_every,R)
    trajs   = Matrix{Union{Missing,Tuple{Int,Int}}}(undef,1+floor(Int,floor(Int,tmax/dt)/save_every),R)
    angle   = Array{Float64}(undef,1+floor(Int,floor(Int,tmax/dt)/save_every),L,L,R)
    omega   = Array{Float64}(undef,L,L,R)
    Threads.@threads for r in 1:R
        thetas = create_isolated_vortex(L,L,+1)
        # omegas = vcat(rand(Normal(Var,0),Int(L/2),L),rand(Normal(-Var,0),Int(L/2),L))
        omega[:,:,r] = omegas = rand(Normal(0,sqrt(Var)),L,L)
        trajs[:,r],angle[:,:,:,r] = evolve_FBC_angle(Q,thetas,omegas,T,dt,tmax,save_every)
    end

    trajs_aux = Matrix{Tuple{Union{Missing,Number},Union{Missing,Number}}}(undef,1+floor(Int,floor(Int,tmax/dt)/save_every),R)
    for i in eachindex(trajs)
        if ismissing(trajs[i]) trajs_aux[i] = (missing,missing)
        else trajs_aux[i] = Float64.(trajs[i])
        end
    end

    return trajs_aux,angle,omega
end

function evolve_FBC_angle(Q::Int,thetas::Matrix{Float64},omegas::Matrix{Float64},T::Number,dt::Float64,tmax::Number,save_every::Int)
    L = size(thetas)[1]
    nsteps_max = floor(Int,tmax/dt)
    history_locations = Vector{Union{Tuple{Int16,Int16},Missing}}(missing,1+floor(Int,nsteps_max/save_every))
    history_angles    = Array{Float64}(undef,1+floor(Int,nsteps_max/save_every),L,L)
    # Initial location
    last_location,~ = spot_single_default_global(thetas)
    history_locations[1]  = last_location
    history_angles[1,:,:] = thetas

    # Time Evolution
    for i in 1:nsteps_max
        thetas = update_FBC(thetas,omegas,L,T,dt) # O(L²)
        if i%save_every == 0
            try
                last_location,alone = spot_single_default_local(thetas,last_location,Q)
                if alone # there are no other vortices in the surroundings (the '2margin' square)
                    saved_location = last_location
                else # there are other vortices in the surroundings that could pollute the data
                    saved_location = missing
                end
                history_locations[1+div(i,save_every)]  = saved_location  # O(1)
                history_angles[1+div(i,save_every),:,:] = thetas
            catch  # we lost our vortex (out of lattice, annihilation)
                printstyled("Warning : Vortex lost, simulation stopped at t = $(dt*i). \n"; color = :yellow)
                break # we leave all the remaining values as they are : missing
            end
        end

    end

    return history_locations,history_angles
end

# function spot_runaways(lattice_omega::Matrix{Float64},threshold::Number,nbroken::Int)::Vector{Tuple{Int,Int}} # return their location (i,j) in a Array of Tuple
#     runaway_locations = Tuple[]
#     L = size(lattice_omega)[1]
#
#     for i in 1:L
#         for j in 1:L
#             omega_spin = lattice_omega[i,j]
#             omega_neighbours = get_omega_nearest_neighbours(lattice_omega,i,j,L)
#             Δω = abs.(omega_neighbours .- omega_spin)
#
#             # if sum(Δω .> threshold) ≥ nbroken
#             if sum(Δω .> threshold) == nbroken
#                 push!(runaway_locations,(i,j))
#             end
#         end
#     end
#     return runaway_locations
# end
#
# function get_omega_nearest_neighbours(omegas::Matrix{Float64},i::Int,j::Int,L::Int)::Vector{Float64}
#     return [omegas[mod1(i+1,L),j], ## Note : mod1 instead of mod to deal with the indexing at from 1
#             omegas[mod1(i-1,L),j],
#             omegas[i,mod1(j+1,L)],
#             omegas[i,mod1(j-1,L)] ]
# end
#
#

function diffops(thetas) # returns grad_sq,div_sq,rotational
    L = size(thetas)[1]
    grad_sq    = fill(NaN,L,L)
    div_sq     = fill(NaN,L,L)
    rotational = fill(NaN,L,L)
    for j in 2:L-1
        for i in 2:L-1
            angle_neighbours = get_angle_nearest_neighbours(thetas,i,j,L) # O(1)
            dθdx = arclength(angle_neighbours[2],angle_neighbours[1])/2
            dθdy = arclength(angle_neighbours[4],angle_neighbours[3])/2
            #= Note : the function arclength handles the problems raised by the
            periodicity of the variable θ. Indeed, when you try to return the
            difference between 0 and 2π-ϵ, since the two vectors are close in
            space, they should not result in a 2π-ϵ difference. The result
            should be -ϵ =#
            θ    = thetas[i,j]
            # grad_sq[i,j]    =( dθdy + dθdx)^2
            grad_sq[i,j]    = sqrt(dθdy^2 + dθdx^2)
            div_sq[i,j]     = cos(θ)*dθdy - sin(θ)*dθdx
            rotational[i,j] = sin(θ)*dθdy + cos(θ)*dθdx
        end
    end
    return grad_sq,div_sq,rotational
end

##
# using Roots,SpecialFunctions
# N = 10:10:10000
# roo = zeros(length(N))
# for n in eachindex(N)
#     f(x) = 1/x^2*erf(x*sqrt(log(N[n])/2))*erf(1/x)^(2N[n]) - 1
#     roo[n] = find_zero(f,(0.01,1))
# end
# plot(roo,xaxis=:log)
# plot(exp.(-roo))

## To keep, to recall the difference between heatmap and quiver
# ttt = reshape([0,pi/2,pi,-pi/3],(2,2))
# heatmap(ttt')
# quiver!(xs,ys,quiver=(vec(cos.(ttt)),vec(sin.(ttt))),label=false,color=:blue,size=(1000,1000))
