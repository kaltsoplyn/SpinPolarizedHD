using QuantumOptics, PlotlyJS, ORCA, Parameters, ZChop, FFTW, Findpeaks

#export đ, be, s, sx, sy, sz, sn, sx1, sy1, sz1, H, D, Bx, Bz, Bx_Default, Bz_Default, ρMEt, S_t, Sx_t, Sz_t, dSdt, fourier, solveME, ResultInfo, getPeaks, scatterPeaks

# ---
# Constants, initial definitions etc.
# ---

# Some physical (Rememeber: the whole notebook is in μs and MHz)
const ns = 1.0E-3
const μs = 1.0
const ms = 1.0E3
const GHz = 1/ns
const MHz = 1/μs
const kHz = 1/ms
const G = 1.0E-4    # 1 Gauss = 0.0001 Tesla
const deg = deg2rad(1)

# Default values for:
# B-field amplitude, frequency, and direction
# initial wavefunction polarization,
# dephasing rates γe, γn, and
# default time-span for solving
@with_kw mutable struct defaults
    Bo = 10G
    fB = 0.
    dB = :z
    dP = :x
    γe = 1MHz
    γn = 0.
    τ = 200ns # pulse width
    to = 0μs  # pulse center
    T = [0:0.1ns:2μs;]
    withContB = true
    withPulsedB = false
end
const đ = defaults()

# magnetic moments
const μe = 88093.0;     # 10⁶ × μe/ħ  (electron magnetic moment in MHz/T)
const μn = 0.0;         # get rid of nuclear magnetic moment to lighten calculations

# electron basis
const be = SpinBasis(1//2) # electron

# spin-½ σ-matrices for electron and proton
const sx = 0.5 * sigmax(SpinBasis(1//2))
const sy = 0.5 * sigmay(SpinBasis(1//2))
const sz = 0.5 * sigmaz(SpinBasis(1//2))

# spin-1 σ-matrices for deuterium nucleus (what's with the 1/3 here? I don't think that's true)
const sx1 = (1/3) * sigmax(SpinBasis(1))
const sy1 = (1/3) * sigmay(SpinBasis(1))
const sz1 = (1/3) * sigmaz(SpinBasis(1))

# define the Atom abstract type, for the H and D to subtype into
abstract type Atom end

@with_kw struct Hydrogen <: Atom
    # nucleus basis
    I = 1//2
    bn = SpinBasis(1//2) # nucleus
    b::CompositeBasis = SpinBasis(1//2) ⊗ SpinBasis(1//2) # composite basis
    
    # Hyperfine constant
    A::Float64 = 2231.17; # this is A/ħ (in MHz) = ¼ × 2π × f_HF, since ω_HF = 4 × A/ħ in hydrogen, and f_HF = 1420.41MHz
    
    # nucleus σ-matrices
    sxn = sx
    syn = sy
    szn = sz    
end

@with_kw struct Deuterium <: Atom
    # nucleus basis
    I = 1//1
    bn = SpinBasis(1) # nucleus
    b::CompositeBasis = SpinBasis(1//2) ⊗ SpinBasis(1) # composite basis
    
    # Hyperfine constant
    A::Float64 = 342.83; # this is A/ħ (in MHz) = ⅙ × 2π × f_HF, since ω_HF = 6 × A/ħ in deuterium, and f_HF = 327.38MHz
    
    # nucleus σ-matrices
    sxn = sx1
    syn = sy1
    szn = sz1    
end

# Instantiate H and D
const H = Hydrogen()
const D = Deuterium()

# generic σ-matrix calling dicts based on input direction (symbol)
s = Dict(:x => sx, :y => sy, :z => sz)
sn(a::Atom) = Dict(:x => a.sxn, :y => a.syn, :z => a.szn);



# Magnetic fields are defined as mutable structs, and there are default Bx and Bz defined
@with_kw mutable struct Bf
    Bo::Real = đ.Bo
    f::Real = đ.fB
    axis::Symbol = đ.dB
    θ::Real = 0.0
    Boff::Real = 0.0
end
Bf(Bo::Real, f::Real = 0.0, axis::Symbol = :z) = Bf(Bo, f, axis, 0.0, 0.0)
Bf(axis::Symbol; Bo::Real = 0, f::Real = 0.0) = Bf(Bo, f, axis, 0.0, 0.0)

Bz_Default = Bf(:z; Bo = 10G, f = 0.0)    # default z-axis m.f. => Bf(10 G, 0 Hz, :z; θ = 0, offset = 0)
Bx_Default = Bf(:x; Bo = 0.0, f = 0.0);   # default x-axis m.f. => Bf( 0 G, 0 Hz, :x; θ = 0, offset = 0)

Bz = Bz_Default;
Bx = Bx_Default;

# Added 19/6/2020: Magnetic field pulses
@with_kw mutable struct Bpulse
    Bo::Real = đ.Bo
    to::Real = đ.to
    τ::Real = đ.τ
    axis::Symbol = đ.dB
    θ::Real = 0.0
    Boff::Real = 0.0
end
Bpulse(Bo::Real, to::Real = 0.0, τ::Real = 200ns, axis::Symbol = :z) = Bpulse(Bo, to, τ, axis, 0.0, 0.0)
Bpulse(axis::Symbol; Bo::Real = 0, to::Real = 0.0, τ::Real = 200ns) = Bpulse(Bo, to, τ, axis, 0.0, 0.0)

BPz_Default = Bpulse(:z; Bo = 10G, to = 0.0, τ = 200ns)    # default z-axis m.f. pulse
BPx_Default = Bpulse(:x; Bo = 0.0, to = 0.0, τ = 200ns);   # default x-axis m.f. pulse

BPz = BPz_Default;
BPx = BPx_Default;


# define a bool function to see if the system is time independent. RELIES on the existence of đ, and Bx and Bz
isTimeIndependent()::Bool = !đ.withPulsedB && ( đ.withContB ? ( Bz.Bo != 0 ? Bz.f == 0 : true ) && ( Bx.Bo != 0 ? Bx.f == 0 : true ) : true )

# ---
# Hamiltonians
# ---

# I⋅S interaction Hamiltonian
Hint(a::Atom) = 2*(2*a.I+1)*a.A*(sx⊗a.sxn + sy⊗a.syn + sz⊗a.szn)                

# the magnetic field: has an amplitude, Bo, and - possibly - a frequency, f, and time dependence, and an offset, Boff
Bt(t::Real, Bo::Real, f::Real, Boff::Real)::Real = Boff + Bo * cos(2π * f * t)

# the magnetic field pulse: has an amplitude, Bo, and a width, τ, and - possibly - a non-zero center, to, and an offset, Boff
Bpulse(t::Real, Bo::Real, to::Real, τ::Real, Boff::Real)::Real = Boff + Bo * exp(-(t-to)^2/τ^2)

offAxis = Dict(:x => :z, :y => :z, :z => :x)
# when B.axis = :x and B.θ ≠ 0, then the off-axis component of the B-field goes into :z
# when B.axis = :z and B.θ ≠ 0, then the off-axis component of the B-field goes into :x
# :y case redundant but defined for completeness | don't use; stay only within the (:x,:z)-plane

# Magnetic Hamiltonian: in the Bf/Bpulse structs a main axis is defined (e.g. :z).
# But there's also - possibly - an angle, B.θ, so then there's also an off-axis component (e.g. along :x)
# So, the total m.f. will then be, e.g. for the oscillating case: B = (Bo cos(2π f t) + Boff) (:z cos(θ) + :x sin(θ))
# ... and the magnetic Hamiltonian will comprise an on-axis and an off-axis part.
# The predefined Bx and Bz fields can independently have such components (each has its own main axis, Bo, f, Boff, and θ)
function HB(a::Atom, B::Bf, t::Real = 0.0)
     -(μe*s[B.axis]⊗one(a.bn) + μn*one(be)⊗sn(a)[B.axis]) * Bt(t, B.Bo, B.f, B.Boff) * cos(B.θ) +
    (-(μe*s[offAxis[B.axis]]⊗one(a.bn) + μn*one(be)⊗sn(a)[offAxis[B.axis]])) * Bt(t, B.Bo, B.f, B.Boff) * sin(B.θ) # Magnetic Hamiltonian -- with B || B.axis, where axis = :x, :y, or :z
end

function HB(a::Atom, B::Bpulse, t::Real = 0.0)   # this is ok >> multiple dispatch creates different HB for types Bf and Bpulse
    -(μe*s[B.axis]⊗one(a.bn) + μn*one(be)⊗sn(a)[B.axis]) * Bpulse(t, B.Bo, B.to, B.τ, B.Boff) * cos(B.θ) +
   (-(μe*s[offAxis[B.axis]]⊗one(a.bn) + μn*one(be)⊗sn(a)[offAxis[B.axis]])) * Bpulse(t, B.Bo, B.to, B.τ, B.Boff) * sin(B.θ) # Magnetic Pulse Hamiltonian -- with B || B.axis, where axis = :x, :y, or :z
end

# Total Hamiltonian implicitly uses the predefined Bx and Bz fields
# So, in the Hamiltonian, and in everything that follows, the magnetic field IS NOT an argument
Ham(a::Atom, t::Real = 0.0) = Hint(a) + convert(Float64, đ.withContB)*(HB(a, Bz, t) + HB(a, Bx, t)) + convert(Float64, đ.withPulsedB)*(HB(a, BPz, t) + HB(a, BPx, t))      # total Hamiltonian assuming a Bx and a Bz m.f.


# ---
# Wavefunctions and density matrices
# ---

function ψe(dir::Symbol = đ.dP) # the electronic spinup state along the three axes
    if dir == :x
        1/√2 * (spinup(be) + spindown(be)) # |e:↑>ₓ
    elseif dir == :y
        1/√2 * (spinup(be) + im * spindown(be))   # |e:↑>ᵧ
    else
        spinup(be)   # |e:↑>z
    end
end

ψp(a::Atom; dir::Symbol = đ.dP)::Ket = ψe(dir)⊗spinup(a.bn)
ψm(a::Atom; dir::Symbol = đ.dP)::Ket = ψe(dir)⊗spindown(a.bn)

ψ0(a::Hydrogen; dir::Symbol = đ.dP)::Ket = 0.0 * spinup(be)⊗spinup(a.bn)  # 0 × |e:whatever>⊗ |n:whatever> >>> this is zero for Hydrogen
ψ0(a::Deuterium; dir::Symbol = đ.dP)::Ket = ψe(dir)⊗Ket(a.bn, [0,1,0])   # |e:↑>⊗ |n:0>

# the initial density matrix for atom a, describing the state [electron at spin-up along direction dir + randomized nucleus]
ρo(a::Atom; dir::Symbol = đ.dP) = (dm(ψp(a; dir = dir)) + dm(ψ0(a; dir = dir)) + dm(ψm(a; dir = dir)))/(2*a.I + 1)  # density matrix |e:↑>ₓ<e:↑|ₓ ⊗ (|n:↑><n:↑| + |n:0><n:0| + |n:↓><n:↓|)

# Jump operators for the master equation (γe, γn in MHz)
function Js(a::Atom; γe::Real = đ.γe, γn::Real = đ.γn)     #=::Array{_A where _A <: SparseOperator, 1}=#
    [γe*sz⊗one(a.bn), γn*one(be)⊗a.szn]; # pure dephasing
end


# ---
# The Master equation solver
# ---

JsDagger(a::Atom; γe::Real = đ.γe, γn::Real = đ.γn) = map(dagger, Js(a; γe = γe, γn = γn))


# Density matrix as solution to master equation
"""
```
    ρMEt( Mandatory args: a::Atom, 
           Optional args: T::Time Array to solve for (1D Real) [default: đ.T];
         Named arguments: dirP::Symbol atomic polarization orientation [default: đ.dP | generally :x, :y, :z] 
                          γe::electron dephasing [default: đ.γe], 
                          γn::nuclear dephasing [default: đ.γn])
```

Calculates a density matrix which solves the master equation, for each time point in the times array, `T`.\n
The initial `ρo` used in the calculations is dictated by the initial atomic polarization argument, `dirP`, and is calculated by the `ρo` function.\n
The `Bx` and `Bz` magnetic field constructs defined earlier are used implicitly in the calculations.\n 
Also, `BPx` and `BPz` are defined for pulsed magnetic fields.
In the defaults struct, đ, two booleans determine if we have continuous (`Bx`, `Bz` >> `đ.withContB`) and/or pulsed (`BPx`, `BPz` >> `đ.withPulsedB`) fields.
If we want the calculations to be performed for different fields than the ones defined, we have to mutate the `Bx` and/or `Bz` (and/or `BPx`, `BPz`) beforehand, e.g.:\n
```
    Bz.Bo = 10G
    Bz.f = 12MHz
    res = ρMEt(H)
```

If `Bz.f` and `Bx.f` are both zero, and `đ.withPulsedB == false`, then the Hamiltonian is time independent, and the regular solver (`timeevolution.master`) is used for the solution.

If `Bz.f` or `Bx.f` is non-zero, or `đ.withPulsedB == true`, then the magnetic field is time dependent, `Bo ̂e cos(2π fB t)`, where ê the direction of the field, 
or `Bo ̂e exp(-(t-to)²/τ²)`, in the case of a - Gaussian - pulsed B-field.
In this case the Hamiltonian is time-dependent, and the dynamic solver is used (`timeevolution.master_dynamic`).

*Notes:*\n 
1) `Bz.Bo` and `Bz.Boff` (DC offset) can be given in Gauss, as e.g. `200G` (`G` is defined as `1E-4`) 
2) The same for the `Bx`, `BPx`, and `BPz` constructs
2) Default values are in the struct `đ` (\\dj), and can be changed as, e.g.: `đ.fB = 100kHz`  or  `đ.dP = :z`

***Returns:***\n
A list of QuantumOptics.jl density matrices solving the master equation for each point of the time array `T`.\n
(so, the density matrix at index `i` in the resulting list, is the solution of the master equation at time `T[i]`)
"""
function ρMEt(a::Atom, T::Array{N,1} = đ.T ; dirP::Symbol = đ.dP, γe::Real = đ.γe, γn::Real = đ.γn) where N <: Real
    ρ0 = ρo(a; dir = dirP)

    if (isTimeIndependent())
        h =  Ham(a, 0.0)
        j = Js(a; γe = γe, γn = γn)
    
        last(timeevolution.master(T, ρ0, h, j #= ; maxiters = 1E8 =#))
    else
        function time_propagation_helper(t, rho)
            return Ham(a, t), Js(a; γe = γe, γn = γn), JsDagger(a; γe = γe, γn = γn)
        end
        
        last(timeevolution.master_dynamic(T, ρ0, time_propagation_helper #= ; maxiters = 1E8 =#))
    end
end;


# ---
# Expectation values and time-derivatives
# ---

# time evolution of the expectation value of sx and sz for the electron
Sx_t(a::Atom, ρ) = real(expect(sx ⊗ one(a.bn), ρ));
Sz_t(a::Atom, ρ) = real(expect(sz ⊗ one(a.bn), ρ));
# ... or of any electronic operator Op_e
"""
```
    S_t(a::Atom, ρ, Op_el)
```

Calculates the expectation value `<Ô> = Tr(ρ.Ô)`, where `Ô = Op_el` is an electronic operator.

*Note:* Use `Sx_t(a, ρ)` or `Sz_t(a, ρ)` directly if `Op_el = sx` or `sz`.

***Arguments***\n
`a::Atom`, the atomic species, `H` or `D`\n
`ρ`, a QuantumOptics.jl density matrix, describing the state upon which we calculate the expectation value\n
`Op_el`, the electronic operator we want to calculate the expectation value of (e.g. `sx`, `sx + im*sy`, `sz` etc.)

***Returns***\n
A number, most probably a Float64; the expectation value of `Ô` across `ρ`.
"""
S_t(a::Atom, ρ, Op_el) = real(expect(Op_el ⊗ one(a.bn), ρ));

# Derivative function
"""
```
    dSdt(timeSeries, timePointsArray) : numeric derivative of a time series, with reference to a list of times.\n
    dSdt(timeSeries, dt)              : numeric derivative of a time series, for time increment dt.\n
```

The `timeSeries` normaly is a list which contains the expectation values of an operator, one for each time point in the `timePointsArray` list.

***Arguments:***\n 
`timeSeries` = time series `<: AbstractArray 1D` ,\n 
`timePointsArray` = time Points Array `<: AbstractArray 1D` of the same length.\n

*or* 

`timeSeries` = time series `<: AbstractArray 1D` ,\n 
`dt` = time increment `<: Real`

***Returns:***\n
An array of the numerical derivatives of the input `timeSeries`, of the same length.
"""
function dSdt(tseries::AbstractArray{N,1}, T::AbstractArray{N,1} = đ.T) where N <: Real
    @assert length(tseries) == length(T) "Error: Arguments of the dSdt function should be of equal length"
    
    dt = T[2] - T[1]
    ddt = [(tseries[i+1] - tseries[i-1])/(2dt) for i in 2:length(T)-1]
    return prepend!(append!(ddt, (tseries[end]-tseries[end-1])/dt), (tseries[2]-tseries[1])/dt)
    # prepend and append half derivatives, so that length doesn't change
end

function dSdt(tseries::AbstractArray{N,1}, dt::M = đ.T[2]-đ.T[1]) where {N,M <: Real}
    ddt = [(tseries[i+1] - tseries[i-1])/(2dt) for i in 2:length(tseries)-1]
    return prepend!(append!(ddt, (tseries[end]-tseries[end-1])/dt), (tseries[2]-tseries[1])/dt)
    # prepend and append half derivatives, so that length doesn't change
end


# ---
# Fourier function
# ---

"""
```
    fourier(timeSeries, timePointsArray)
```

Produces the fft of a time series, with reference to a list of times.\n

***Arguments:***\n
timeSeries = time series <: AbstractArray 1D , t = timePointsArray <: AbstractArray 1D of the same length.\n

***Returns:***\n 
A tuple of two arrays, (frequency range, Fourier list), both of half the length of the input time series (due to Fourier mirroring)
"""
function fourier(tseries::AbstractArray{N,1}, T::AbstractArray{M,1} = đ.T) where {N,M <: Real}
    @assert length(tseries) == length(T) "Error: Arguments of the fourier function should be of equal length"
    
    dt = T[2] - T[1]                  # time increment of time series
    fmax = (1/dt)/2                   # max sampled frequency = sampling rate / 2 
    df = 1/(T[end]-T[1])              # frequency resolution
    len = Integer(floor(length(T)/2)) # 'floor' to account for possible odd length(T)
    _frng = LinRange(0, fmax, len)    # construct the frequency range array
    _fft = abs.(fft(tseries)[1:len])  # construct the - abs - of the Fourier of the input time series
    return _frng, _fft
    end;

    
# ---
# Lorentz 3-points trick definitions
# ---
"""
``` 
    foL(f1, f2, f3, y1, y2, y3)
    foL([f1, f2, f3], [y1, y2, y3])
```

Finds the frequency at the peak of a Lorentz distribution, given three data points, `(f1, y1)`, `(f2, y2)`, `(f3, y3)`. 

The idea is to pick the max-y point `(f2, y2)` from a discrete list of data points encoding a lorentzian peak, and its two adjacent points to the left and right,
and approximate the peak frequency with better precision than the frequency bin width `df = f_i+1 - f_i` 
"""
function foL(x1::N1,x2::N2,x3::N3,y1::N4,y2::N5,y3::N6)::Float64 where {N1,N2,N3,N4,N5,N6 <: Real}
    0.5*(x1^2*y1*(y2-y3) + x2^2*y2*(y3-y1) + x3^2*y3*(y1-y2))/((x1*y1*(y2-y3) + x2*y2*(y3-y1) + x3*y3*(y1-y2)))
end
function foL(xs::Array{N1,1},ys::Array{N2,1})::Float64  where {N1,N2 <: Real}
    0.5*(xs[1]^2*ys[1]*(ys[2]-ys[3]) + xs[2]^2*ys[2]*(ys[3]-ys[1]) + xs[3]^2*ys[3]*(ys[1]-ys[2]))/((xs[1]*ys[1]*(ys[2]-ys[3]) + xs[2]*ys[2]*(ys[3]-ys[1]) + xs[3]*ys[3]*(ys[1]-ys[2])))
end

"""
```
    aoL(f1, f2, f3, y1, y2, y3)
    aoL([f1, f2, f3], [y1, y2, y3])
```

Finds the amplitude at the peak of a Lorentz distribution, given three data points, `(f1, y1)`, `(f2, y2)`, `(f3, y3)`.

See the documentation for `foL`.

Here, it is implicitly assumed that `df = f3 - f2 = f2 - f1`. 
"""
function aoL(x1::N1,x2::N2,x3::N3,y1::N4,y2::N5,y3::N6)::Float64 where {N1,N2,N3,N4,N5,N6 <: Real}
    (8*y1*y2*y3*(x1*y1*(y2-y3) + x3*(y1-y2)*y3 + x2*y2*(-y1+y3)))/((x2-x1)*(y1^2*(y2-4*y3)^2 + y2^2*y3^2-2*y1*y2*y3*(y2+4*y3)))
end
function aoL(xs::Array{N1,1},ys::Array{N2,1})::Float64  where {N1,N2 <: Real}
    (8*ys[1]*ys[2]*ys[3]*(xs[1]*ys[1]*(ys[2]-ys[3]) + xs[3]*(ys[1]-ys[2])*ys[3] + xs[2]*ys[2]*(-ys[1]+ys[3])))/((xs[2]-xs[1])*(ys[1]^2*(ys[2]-4*ys[3])^2 + ys[2]^2*ys[3]^2-2*ys[1]*ys[2]*ys[3]*(ys[2]+4*ys[3])))
end;



# ---
# Main solver definitions
# ---
"""
```
    solveME(Mandatory args: a::Atom, 
             Optional args: T::Array of times to solve for (1D Real) [default: đ.T];
           Named arguments: Op_el the electronic operator to solve for (expectation value >> dS/dt >> Fourier) [default: sx]
                            dirP::Symbol :x, :y, :z atomic polarization orientation [default: đ.dP]
                            γe::electron dephasing [default: đ.γe], 
                            γn::nuclear dephasing [default: đ.γn])
```

Solves the master eq and returns the FFT of the `d<Op_el>/dt` (tupled with the list of frequencies).

*Note:* Internally, the `ρMEt` function is called, which uses the pre-defined magnetic fields `Bx` and/or `Bz` and/or `BPx` and/or `BPz`.
(dependent on the values of the controlling booleans `đ.withContB` and `đ.withPulsedB`) 

***Returns:*** a tuple (frequencies list, Fourier list), both 1D arrays of Float64, `Array{Float64, 1}`
"""
@inline function solveME(a::Atom, T::Array{N,1} = đ.T ; Op_el = sx, dirP::Symbol = đ.dP, γe::Real = đ.γe, γn::Real = đ.γn) where N <: Real
    # Solve master eq.
    _ρt = ρMEt(a,T; dirP = dirP, γe = γe, γn = γn);
    # Find exp val of sx and its time derivative
    _st = S_t(a, _ρt, Op_el)
    _dsdt = dSdt(_st, T)
    # Fourier of ds/dt
    fourier(_dsdt, T)
end

# The helper functions below are used to ensure that the peaks lists will have the appropriate length for H or D
# FOR TRANSVERSE FIELD ONLY i.e. 4 for H, and 6 for D, even for tiny fields where some of the peaks are invisible
# However, this notebook has become miore generic, so these functions need revision
# 2020, May 26: Right now they are used only when f_Bx = f_Bz = 0
function getPeaks_helper(a::Hydrogen, pks)
    _peaks = pks
    length(_peaks) > 4 ? (return sort(_peaks[1:4])) : _peaks = sort(_peaks)
    # the following if/elseif block is an attempt to make sure the function always outputs 4 peaks (2@low-f & 2@high-f)
    # For B ~ 0 >>> only one or two peaks >>> output indices = [1, 1, peak #1 index, peak #1 or #2 index]
    # For small B >>> 3 peaks (low-f peaks not resolved) >>> repeat low-f >>> output = [peak #1, peak #1, peak #2, peak #3]
    if length(_peaks) == 1 # i.e. B = 0
        _peaks = [1, 1, _peaks[1], _peaks[1]]
    elseif length(_peaks) == 2 # very small B (high-f resolved, but no low-f peaks)
        _peaks = [1, 1, _peaks[1], _peaks[2]]
    elseif length(_peaks) == 3 # i.e. small B where the low peaks are detected as one
        prepend!(_peaks, _peaks[1])
    end
    return _peaks
end

function getPeaks_helper(a::Deuterium, pks)
    _peaks = pks
    length(_peaks) > 6 ? (return sort(_peaks[1:6])) : _peaks = sort(_peaks)
    # the following if/elseif block is an attempt to make sure the function always outputs 6 peaks (3@low-f & 3@high-f)
    # see similar discussion in Hydrogen case
    if length(_peaks) == 1 # i.e. B = 0
        _peaks = [1, 1, 1, _peaks[1], _peaks[1], _peaks[1]]
    elseif length(_peaks) == 2 # very small B (high-f resolved, but no low-f peaks)
        _peaks = [1, 1, 1, _peaks[1], _peaks[1], _peaks[2]]
    elseif length(_peaks) == 3 # very small B (high-f resolved, but no low-f peaks)
        _peaks = [1, 1, 1, _peaks[1], _peaks[2], _peaks[3]]
    elseif length(_peaks) == 4 # very small B (high-f resolved, but no low-f peaks)
        _peaks = [_peaks[1], _peaks[1], _peaks[1], _peaks[2], _peaks[3], _peaks[4]]
    elseif length(_peaks) == 5 # very small B (high-f resolved, but no low-f peaks)
        _peaks = [_peaks[1], _peaks[1], _peaks[2], _peaks[3], _peaks[4], _peaks[5]]
    end
    return _peaks
end


# A couple helper functions more
# Op2Sym converts an operator matrix sx, sy, sz, to the corresponding axis symbol, :x, :y, :z
# If the operator does not match one of the sx, sy, sz matrices, it returns :other_operator
# However, other operators are accepted for the calculations (they are just not identified by this function)
Op2Sym(Op_el)::Symbol = Op_el == sx ? :sx : (Op_el == sy ? :sy : (Op_el == sz ? :sz : :other_operator))

"""
`ResultInfo` is a struct containing information about the results of the calculation.\n

The fields are: \n
`Operator::Symbol` : for which operator the calculations were performed\n
`AtomicPolarization`::Symbol : along which axis are the atoms polarized initially\n
`HadContinuousB` ::Bool : whether a continous B (constant or oscillatory) was present, i.e. `Bx` and/or `Bz`\n
`Bz::Bf` obj : the Bz field details\n
`Bx::Bf` obj : the Bx field details\n
`HadPulsedB` ::Bool : whether a pulsed B was present, i.e. `BPx` and/or `BPz`\n
`BPz::Bpulse` obj : the BPz field details\n
`BPx::Bpulse` obj : the BPx field details\n
`PeakInds::[Int64]`: List of indices (positions) in the FFT list where the peaks appear\n
`PeakFreqs::[F64]`  : List of peak frequencies (corresponding to peaks from higher to lower)\n
`PeakAmps::[F64]`  : List of peak amplitudes\n
`PeakFreqsLorentz::[F64]`  : List of peak frequencies calculated with the Lorentz trick from the FFT data\n
`PeakAmpsLorentz::[F64]`  : List of peak amplitudes calculated with the Lorentz trick from the FFT data\n
"""
struct ResultInfo
    Operator::Symbol
    AtomicPolarization::Symbol
    HadContinuousB::Bool
    Bz::Bf
    Bx::Bf
    HadPulsedB::Bool
    BPz::Bpulse
    BPx::Bpulse
    PeakInds::Array{Int64,1}
    PeakFreqs::Array{Float64,1}
    PeakAmps::Array{Float64,1}
    PeakFreqsLorentz::Array{Float64,1}
    PeakAmpsLorentz::Array{Float64,1}
end

function Base.show(io::IO, res::ResultInfo)
    compact = get(io, :compact, false)

    if !compact
        println("Results for current parameters:")
        println("===============================")
        for field in fieldnames(ResultInfo)
            if (res.HadContinuousB == false && (field == :Bx || field == :Bz))
                continue
            end
            if (res.HadPulsedB == false && (field == :BPx || field == :BPz))
                continue
            end
            l = length(string(field))
            print("$field $(l > 10 ? "\t" : (l < 6 ? "\t\t\t" : "\t\t")): $(getfield(res, field)) \n")
        end
    end
end


# below, the main function returning results is defined
"""
`getPeaks(a::Atom, T::Array{N,1} = đ.T ; Op_el = sx, dirP::Symbol = đ.dP, γe::Real = đ.γe, γn::Real = đ.γn)::ResultInfo where N <: Real`

Returns a `ResultInfo` struct with the results of the calculation.\n
*Note:* Internally, the `ρMEt` function is called, which uses the pre-defined magnetic fields `Bx` and `Bz`.
If different specs are required from the magnetic field, the field will have to be mutated beforehand, e.g.:\n
```
    Bz.Bo = 10G
    Bz.f = 12MHz
    Bx.Bo = 0.5G
    Bx.f = 0.0
    res = getPeaks(H)
```

***Arguments:***\n
*Mandatory*\n
`a::Atom`, the atomic system to solve for, `H` or `D` (these structs have been defined in the Constants etc. section)

*Optional*  \n
`T::Array{Real, 1}`, the array of times to solve for [default `đ.T`]

*Keyword*\n  
`Op_el`, the electronic operator for which the calculations are performed [default `sx`]\n
`dirP::Symbol`, along which axis are the atoms polarized initially [default `đ.dP` | generally `:x`, `:y`, `:z`]\n
`γe::Real`, the electronic polarization dephasing rate [default `đ.γe` | *note:* `γe = 1`, means 1 MHz]\n
`γn::Real`, the nuclear polarization dephasing rate [default `đ.γn` | *note:* `γn = 1`, means 1 MHz]\n


***Returns*** \n 
A `ResultInfo` struct (see `?ResultInfo` for extra help).
"""
function getPeaks(a::Atom, T::Array{N,1} = đ.T ; Op_el = sx, dirP::Symbol = đ.dP, γe::Real = đ.γe, γn::Real = đ.γn)::ResultInfo where N <: Real
    frng, FFT = solveME(a, T; Op_el = Op_el, dirP = dirP, γe = γe, γn = γn)
    
    # find indices of peaks
    peaks = findpeaks(FFT)
    
    if isTimeIndependent()  # for DC fields we know how many peaks we're expecting, so we account for them all, even if the B-field is small and not all peaks are visible yet
        peaks = getPeaks_helper(a, peaks)
        else # for AC fields, sidebands may be all over the place, so put some limits to what the accepted peak prominence is
        peaks = findpeaks(FFT; minProm = 0.01*maximum(FFT))
    end
    
    # then we pinpoint the frequencies better (?) with the Lorentz trick
    peakFreqsLorentz = rand(length(peaks))
    peakAmpsLorentz = rand(length(peaks))
    for (i, pk) in enumerate(peaks)
        if pk == 1
            peakFreqsLorentz[i] = 0.0
            peakAmpsLorentz[i] = 0.0
        else
            tripletFs = frng[pk-1:pk+1]
            tripletAs = FFT[pk-1:pk+1]
            peakFreqsLorentz[i] = foL(tripletFs..., tripletAs...)
            peakAmpsLorentz[i] = aoL(tripletFs..., tripletAs...)
        end
    end
    
    # return Dict of results
    return ResultInfo(Op2Sym(Op_el), dirP, đ.withContB, Bz, Bx, đ.withPulsedB, BPz, BPx, peaks, frng[peaks], FFT[peaks], peakFreqsLorentz, peakAmpsLorentz)
end


# ---
# Plot helpers
# ---
"""
  `scatterPeaks(peaks::ResultInfo)` \n
  `scatterPeaks(Lorentz_freqs_List, Lorentz_amps_List)` \n

Prepares the Lorentz peaks of the Fourier of the `d<S>/dt` to be plotted. \n
***arguments:*** `pks`:: a `ResultInfo` returned from the getPeaks function, from which the Lorentz freqs and amps are extracted\n
or two lists, `[frequencies list]` & `[amplitudes list]` directly.

Plot directly with `plot(scatterPlots(arg))` or superimpose on another plot with `addtraces!(existingPlot, scatterPeaks(arg))`
"""
function scatterPeaks(pks::ResultInfo)
    scatter(;x=(pks.PeakFreqsLorentz)/1000, y=pks.PeakAmpsLorentz, name="Lorentz peaks", mode = "markers", marker=attr(color=:red, size=8, symbol="circle-open"))
end;

function scatterPeaks(freqs::Array{N,1}, amps::Array{M,1}) where {M,N <: Real}
    @assert length(freqs) == length(amps) "Error: the two supplied lists should be of equal length"
    scatter(;x=freqs/1000, y=amps, name="Lorentz peaks", mode = "markers", marker=attr(color=:red, size=8, symbol="circle-open"))
end