import numpy as np
from scipy.optimize import fmin_ncg, fmin_powell, fmin_cg
from matplotlib import colors, ticker, cm
from math import sqrt, ceil, pi, cos, sin
from cmath import exp
from numpy import random
import threading

############
# PARAMAETRS
############

# Let's fix an area A
tt = 2
A = tt * 100

# Let's fix the number of discretization points, which will be the same amount in X and Y
N_x = 150
N_y = N_x

# Let's fi the Mingarelli phases
α_x = 0
α_y = 0
γ_x = α_x
γ_y = α_y

# Healing length
ξ = 0.05

# Interaction parameter
α = 0.1
W_0 = 0.5 / ξ**2
W_2 = α * W_0

# Anistropy
τ = 0.0

# Number of vortices per unit cell 
q = 1 * tt
Ω = np.pi*q / (N_x*N_y)

# Chemical pontials
μ_a = W_0
μ_b = W_0

# Switch on/off spinorness: on=1 and off=0
on_spinor = 1

# Write out wave functions: yes=1 and no=0
on_wave = 1

if on_wave == 1:
	path = '_wavefunctions'
else:
	path = ''

# Convergence precision
prec = 1e-5

# Range aspect ratio of the unit cell: R
rlist =np.arange(1.0, 2.0+0.0001, 0.1)
nr = len(rlist)
rmin = min(rlist)
rmax = max(rlist)

##############################################
# CONSTRUCT THE LOSS FUNCTION AND ITS GRADIENT
##############################################

# Below we define the phases for the twisted boundary conditions

# Phases from the TBC for first component of the spinor

TBC_x_plus_a = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  i == N_x-1:
             TBC_x_plus_a[i][j] = np.exp(1j * (Ω*N_x*j + α_x))
        else:
             TBC_x_plus_a[i][j] = 1.0

TBC_x_minus_a = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  i == 0:
             TBC_x_minus_a[i][j] = np.exp(-1j * (Ω*N_x*j + α_x))
        else:
             TBC_x_minus_a[i][j] = 1.0

TBC_y_plus_a = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  j == N_y-1:
             TBC_y_plus_a[i][j] = np.exp(-1j * (Ω*N_y*i + α_y))
        else:
             TBC_y_plus_a[i][j] = 1.0

TBC_y_minus_a= np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  j == 0:
             TBC_y_minus_a[i][j] = np.exp(1j * (Ω*N_y*i + α_y))
        else:
             TBC_y_minus_a[i][j] = 1.0

# Phases from the TBC for second component of the spinor

TBC_x_plus_b = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  i == N_x-1:
             TBC_x_plus_b[i][j] = np.exp(1j * (Ω*N_x*j + γ_x))
        else:
             TBC_x_plus_b[i][j] = 1.0

TBC_x_minus_b = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  i == 0:
             TBC_x_minus_b[i][j] = np.exp(-1j * (Ω*N_x*j + γ_x))
        else:
             TBC_x_minus_b[i][j] = 1.0

TBC_y_plus_b = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  j == N_y-1:
             TBC_y_plus_b[i][j] = np.exp(-1j * (Ω*N_y*i + γ_y))
        else:
             TBC_y_plus_b[i][j] = 1.0

TBC_y_minus_b = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
        if  j == 0:
             TBC_y_minus_b[i][j] = np.exp(1j * (Ω*N_y*i + γ_y))
        else:
             TBC_y_minus_b[i][j] = 1.0

# Phases from the vector potential

A_x_plus = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
             A_x_plus[i][j] = np.exp(-1j * Ω * j)

A_x_minus = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
             A_x_minus[i][j] = np.exp(1j * Ω * j)

A_y_plus = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
             A_y_plus[i][j] = np.exp(1j * Ω * i)

A_y_minus = np.zeros(shape=(N_x, N_y)) + 1j * np.zeros(shape=(N_x, N_y))

for i in range(N_x):
      for j in range(N_y):
             A_y_minus[i][j] = np.exp(-1j * Ω * i)

# Putting things together

U_x_plus_a = TBC_x_plus_a * A_x_minus
U_x_minus_a = TBC_x_minus_a * A_x_plus
U_y_plus_a = TBC_y_plus_a * A_y_minus
U_y_minus_a = TBC_y_minus_a * A_y_plus

U_x_plus_b = TBC_x_plus_b * A_x_minus
U_x_minus_b = TBC_x_minus_b * A_x_plus
U_y_plus_b = TBC_y_plus_b * A_y_minus
U_y_minus_b = TBC_y_minus_b * A_y_plus

# Define the loss function

def energy(Ψ_1d, N_x, N_y, μ_a, μ_b, W_0, W_2, τ, r, A):

    ax = sqrt((A/(N_x*N_y)) * r)

    ay = sqrt((A/(N_x*N_y)) / r)

    w_x = 1 / ax**2

    w_y = 1 / ay**2

    Ψ_a = Ψ_1d[0 :N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d[N_x * N_y : 2 * N_x * N_y].reshape((N_x, N_y))

    Ψ_b = Ψ_1d[2 * N_x * N_y : 3 * N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d[3 * N_x * N_y : 4 * N_x * N_y].reshape((N_x, N_y))

    # Kinetic energy

    E_0_a = np.sum(w_x * np.abs(np.roll(Ψ_a, -1, axis = 0) * U_x_plus_a  - Ψ_a)**2 + w_y * np.abs(np.roll(Ψ_a, -1, axis = 1) * U_y_plus_a  - Ψ_a)**2) / 2

    E_0_b = on_spinor * np.sum(w_x * np.abs(np.roll(Ψ_b, -1, axis = 0) * U_x_plus_b  - Ψ_b)**2 + w_y * np.abs(np.roll(Ψ_b, -1, axis = 1) * U_y_plus_b  - Ψ_b)**2) / 2

    # Chemical potential

    E_μ_a = -μ_a * np.sum(np.abs(Ψ_a)**2)
    
    E_μ_b = -μ_b * np.sum(np.abs(Ψ_b)**2)

    # Interactions

    E_int_02 = (1 / (ax*ay)) * 0.5 * (W_0+W_2) * np.sum((np.abs(Ψ_a)**2 + np.abs(Ψ_b)**2)**2)
    
    E_int_2 = -(1 / (ax*ay)) * 0.5 * W_2 * np.sum(np.abs(Ψ_a**2 + Ψ_b**2)**2)

    # Anisotropy    

    E_τ = -τ * np.sum(np.abs(Ψ_a)**2)

    return E_0_a + E_0_b + E_μ_a + E_μ_b + E_int_02 + E_int_2 + E_τ

# Define the gradient of the loss function

def energy_gradient(Ψ_1d, N_x, N_y, μ_a, μ_b, W_0, W_2, τ, r, A): 

    # Some care required here as we have to convert the complex gradient into real and imaginary parts! 

    ax = sqrt((A/(N_x*N_y)) * r)

    ay = sqrt((A/(N_x*N_y)) / r)

    w_x = 1 / ax**2

    w_y = 1 / ay**2

    Ψ_a = Ψ_1d[0 : N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d[N_x * N_y : 2 * N_x * N_y].reshape((N_x, N_y))

    Ψ_b = Ψ_1d[2 * N_x * N_y : 3 * N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d[3 * N_x * N_y : 4 * N_x * N_y].reshape((N_x, N_y))

    # Kinetic energy

    EG_0_a = (w_x * (2 * Ψ_a - np.roll(Ψ_a,-1, axis = 0) * U_x_plus_a - np.roll(Ψ_a, 1, axis = 0) * U_x_minus_a) +
                         w_y * (2 * Ψ_a - np.roll(Ψ_a,-1, axis = 1) * U_y_plus_a - np.roll(Ψ_a, 1, axis = 1) * U_y_minus_a)) / 2

    EG_0_b = on_spinor * (w_x * (2 * Ψ_b - np.roll(Ψ_b,-1, axis = 0) * U_x_plus_b - np.roll(Ψ_b, 1, axis = 0) * U_x_minus_b) +
                         w_y * (2 * Ψ_b - np.roll(Ψ_b,-1, axis = 1) * U_y_plus_b - np.roll(Ψ_b, 1, axis = 1) * U_y_minus_b)) / 2

    # Chemical potential

    EG_μ_a = -μ_a * Ψ_a

    EG_μ_b = -μ_b * Ψ_b

    # Interactions

    EG_int_a = (1/(ax*ay)) * ((W_0+W_2) * Ψ_a * (np.abs(Ψ_a)**2 + np.abs(Ψ_b)**2) - W_2 * np.conjugate(Ψ_a) * (Ψ_a**2 + Ψ_b**2))

    EG_int_b = (1/(ax*ay)) * ((W_0+W_2) * Ψ_b * (np.abs(Ψ_a)**2 + np.abs(Ψ_b)**2) - W_2 * np.conjugate(Ψ_b) * (Ψ_a**2 + Ψ_b**2))

    # Anisotropy

    EG_τ_a = -τ * Ψ_a

    # Put things together

    EG_a = np.ravel(EG_0_a + EG_μ_a + EG_int_a + EG_τ_a)

    EG_b = np.ravel(EG_0_b + EG_μ_b + EG_int_b)

    # Concatenate real and imaginary parts

    EG_a_separated = np.concatenate((2 * np.real(EG_a), 2 * np.imag(EG_a)))

    EG_b_separated = np.concatenate((2 * np.real(EG_b), 2 * np.imag(EG_b)))

    return np.concatenate((EG_a_separated,EG_b_separated))

# Integration error

def Er(Ψ_1d, N_x, N_y, μ_a, μ_b, W_0, W_2, τ, r, A):

    ax = sqrt((A/(N_x*N_y)) * r)

    ay = sqrt((A/(N_x*N_y)) / r)

    w_x = 1 / ax**2

    w_y = 1 / ay**2

    Ψ_a = Ψ_1d[0 : N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d[N_x * N_y : 2 * N_x * N_y].reshape((N_x, N_y))

    Ψ_b = Ψ_1d[2 * N_x * N_y : 3 * N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d[3 * N_x * N_y : 4 * N_x * N_y].reshape((N_x, N_y))

    # Kinetic energy

    E_0_a = (w_x * np.abs(np.roll(Ψ_a, -1, axis = 0) * U_x_plus_a  - Ψ_a)**2 + w_y * np.abs(np.roll(Ψ_a, -1, axis = 1) * U_y_plus_a  - Ψ_a)**2) / 2

    E_0_b = on_spinor * (w_x * np.abs(np.roll(Ψ_b, -1, axis = 0) * U_x_plus_b  - Ψ_b)**2 + w_y * np.abs(np.roll(Ψ_b, -1, axis = 1) * U_y_plus_b  - Ψ_b)**2) / 2

    # Chemical potential

    E_μ_a = -μ_a * (np.abs(Ψ_a)**2)
    
    E_μ_b = -μ_b * (np.abs(Ψ_b)**2)

    # Interactions

    E_int_02 = (1/(ax*ay)) * 0.5 * (W_0+W_2) * np.sum((np.abs(Ψ_a)**2 + np.abs(Ψ_b)**2)**2)

    E_int_2 = -(1/(ax*ay)) * 0.5 * W_2 * np.sum(np.abs(Ψ_a**2 + Ψ_b**2)**2)

    # Rabi    

    E_τ = -τ * np.sum(np.abs(Ψ_a)**2) 
    
    E_density = (E_0_a + E_0_b + E_μ_a + E_μ_b + E_int_02 + E_int_2 + E_τ) / (ax*ay)
    
    Epx = np.zeros(shape=(N_x,N_y)) 
    Epy = np.zeros(shape=(N_x,N_y)) 

    for i in range(N_x):
          for j in range(N_y):
                 if  i == 0:
                    Epx[i][j] = (E_density[i+1][j]-E_density[i][j]) / ax
                 else:
                    Epx[i][j] = (E_density[i][j]-E_density[i-1][j]) / ax 
                    
    for i in range(N_x):
          for j in range(N_y):
                 if  j == 0:
                    Epy[i][j] = (E_density[i][j+1]-E_density[i][j]) / ay
                 else:
                    Epy[i][j] = (E_density[i][j]-E_density[i][j-1]) / ay               

    qa = (1/(N_x*N_y)) * np.sum(Epx)
    qb = (1/(N_x*N_y)) * np.sum(Epy)
    Error = -(A/2) * (ax*qa + ay*qb)
                    
    return Error

##############
# OPTIMIZATION
##############

blist = np.zeros(nr)
elist = np.zeros(nr)
clist = np.zeros(nr)
nlist = np.zeros(nr)
wlist = np.zeros(shape=(nr, 4*N_x*N_y))

# Parallelization
 
Ψ_1d_out = list()

def leonardo(r):

     # Initial guess

     Ψ_1d_inn = random.rand(4*N_x*N_y)
     Ψ_1d_in = Ψ_1d_inn / sqrt(np.sum(Ψ_1d_inn**2))

     # Optimization

     leo_out = fmin_cg(energy, Ψ_1d_in, energy_gradient, args=(N_x, N_y, μ_a, μ_b, W_0, W_2, τ, r, A), gtol=prec, full_output=True, disp=False)

     Ψ_1d_out.append([r, leo_out])

threads = list()

def paralell_process(list_of_inputs):

        for in_val in list_of_inputs:
                t = threading.Thread(target=leonardo, args=(in_val,))
                threads.append(t)
                t.start()

        for t in threads:
                t.join()

paralell_process(rlist)

# Rearrange outputs of the parallelization, so the list of R are properly arranged

for ia in range(nr):

	for ib in range(nr):

		if rlist[ia] == Ψ_1d_out[ib][0]:
    
			for j in range(4*N_x*N_y):

				if j == 0:

					elist[ia] = Ψ_1d_out[ib][1][1]

					clist[ia] = Ψ_1d_out[ib][1][4]

					wlist[ia][j] = Ψ_1d_out[ib][1][0][j]

					blist[ia] = Er(Ψ_1d_out[ib][1][0], N_x, N_y, μ_a, μ_b, W_0, W_2, τ, Ψ_1d_out[ib][0], A)

					Ψ_a = Ψ_1d_out[ib][1][0][0 : N_x * N_y].reshape((N_x, N_y))+ 1j * Ψ_1d_out[ib][1][0][N_x * N_y : 2 * N_x * N_y].reshape((N_x, N_y))

					nlist[ia] = np.sum(np.abs(Ψ_a)**2) / (N_x*N_y)
				
				else:

					wlist[ia][j] = Ψ_1d_out[ib][1][0][j]

wlist_1d = np.ravel(wlist)

#########
# LOGGING
#########

with open("../logging/Elist" + "_A" + str(A) + "_N" + str(N_x) +"_ξ" + str(ξ) + "_α" + str(α) + "_τ" + str(τ)[:6] + "_q" + str(q) + "_rmin" + str(rmin)[:6] + "_rmax" + str(rmax)[:6] + "_prec" + str(prec)+"_spinspin"+path+".csv","w") as out_file:


    out_string = str(A)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(N_x)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(μ_a)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(μ_b)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(W_0)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(α)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(τ)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(q)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(nr)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(prec)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(α_x)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(α_y)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(γ_x)
    out_string += "\n"
    out_file.write(out_string)

    out_string = str(γ_y)
    out_string += "\n"
    out_file.write(out_string)

    for i in range(nr):
      out_string = ""
      out_string += str(clist[i])
      out_string += "\n"
      out_file.write(out_string)

    for i in range(nr):
      out_string = ""
      out_string += str(elist[i])
      out_string += "\n"
      out_file.write(out_string)

    for i in range(nr):
      out_string = ""
      out_string += str(blist[i])
      out_string += "\n"
      out_file.write(out_string)

    for i in range(nr):
      out_string = ""
      out_string += str(nlist[i])
      out_string += "\n"
      out_file.write(out_string)

    for i in range(nr):
      out_string = ""
      out_string += str(rlist[i])
      out_string += "\n"
      out_file.write(out_string)

    if on_wave == 1:

	    for i in range(len(wlist_1d)):
       		out_string = ""
        	out_string += str(wlist_1d[i])
        	out_string += "\n"
        	out_file.write(out_string)

