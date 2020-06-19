# Spin Polarized Hydrogen and Deuterium using Julia

This repo is all about a Julia base file (SPBase.jl) which can be included in Jupyter notebooks or in other Julia files, and which leverages
the nice Julia packages QuantumOptics.jl, FFTW.jl etc. to study the time evolution of an initially polarized H or D gas, in the presence of 
dephasing and external magnetic fields.  

The electronic polarization is created by photodissociating hydrohalides with sub-ns pulses. After its creation, the polarization is transferred 
back and forth between the electron and the nucleus, at the characteristic HF frequency, and, it decays due to depolarizing collisions etc. 
In the presence of an external magnetic field, precession also occurs. So, can this act as a magnetometer?  

The magnetic fields can form any angle relative to the electronic polarization of the vapor, 
and they can be static (DC), oscillatory, and/or just a fast magnetic field pulse. 
Since this work is particular to our experiments, where the oscillating electronic polarization is detected via the electromotive force 
it induces to a tiny pick-up coil, the observable here is essentially the time-derivative of the expectation value of the electronic 
polarization along the axis of the coil.  

To calculate all this, we solve the time-dependent master equation for a given temporal window, at a specific time-step.
From there, having calculated the density matrix for each time value, we can calculate the expectation value of any observable vs time.

So, we calculate the time derivative of the expectation value of the spin polarization, and, in its FFT spectrum, 
a rich multitude of behaviours owing to the magnetic fields becomes apparent.