import numpy as np
from scipy.integrate import solve_ivp

def derivs(vars, ax, ay): #returns the right hand side of the differential equations.
    return np.array([vars[2], vars[3], ax, ay])

#I used this https://math.stackexchange.com/questions/598685/point-deflecting-off-of-a-circle
#to calculate the change in velocity after a collision. I also normalise the position vector
#to make sure the function doesn't get called more than once for the same collision.
def reflection(vars, Restitution): 
    vx_final = vars[2]-2*vars[0]**2*vars[2]-2*vars[0]*vars[1]*vars[3]
    vy_final = vars[3]-2*vars[1]**2*vars[3]-2*vars[0]*vars[1]*vars[2]
    vx_final = Restitution*vx_final
    vy_final = Restitution*vy_final

    normalisation_factor = 1/np.sqrt(vars[0]**2+vars[1]**2)
    x_final = vars[0]*normalisation_factor
    y_final = vars[1]*normalisation_factor
    
    return np.array([x_final, y_final, vx_final, vy_final])

def BounceFunction(vars, dt, Tmax, Restitution, ax, ay, Max_Bounce):
    Bounce_count, t, g = 0, 0, 9.82
    ax, ay = 0, -g
    
    vars_save, t_save = [[], [], [], []], []

    while t <= Tmax: #I use euler integration until Tmax  or Max_Bounce is reached
        t_save.append(t)
        vars = vars + derivs(vars, ax, ay)*dt
        t += dt

        for i in range(len(vars)): #I save the variables in an array
            vars_save[i].append(vars[i])
        if np.sqrt(vars[0]**2 + vars[1]**2) >= 1: #If the ball hits the wall, it reflects
            vars = reflection(vars, Restitution)
            Bounce_count += 1
        if Bounce_count == Max_Bounce:
            break
        
    return vars_save, t_save

def every_nth(nums, nth): #I use this to cut down on the amount of elements in my arrays. It returns every nth element of an array
    return nums[nth - 1::nth]

def new_derivs(t, vars, acceleration): #These return the same derivatives, just not in a numpy array, so i can use solve_ivp
    return vars[2], vars[3], acceleration[0], acceleration[1]

def cross_boundary(t, vars, acceleration): #This is the event function for solve_ivp, it stops the integration when the ball hits the wall
    return np.sqrt(vars[0]**2 + vars[1]**2) - 1
cross_boundary.terminal = True

#This is the same as BounceFunction, but it uses solve_ivp instead of euler integration
def new_BounceFunction(vars, dt, Tmax, ax, ay, Restitution, Max_Bounce):
    acceleration = [ax, ay]
    vars_save = [[], [], [], []]
    variables = vars
    t_save = np.array([0])
    
    for i in range(Max_Bounce): #Since solve_ivp only does one bounce, i need to call it multiple times
        sol = solve_ivp(new_derivs, (0, Tmax), variables, events = cross_boundary, 
                    max_step = dt, args = (acceleration,))
        t_save = np.append(t_save, np.asarray(sol.t)+ t_save[-1]+dt)#I save the time in an array
        if t_save[-1] >= Tmax: #And check if it has reached Tmax
            break       
        
        #It doesn't take very long to run, so i just save the trajectories in an array
        for j in range(len(sol.y)):
            for k in range(len(sol.y[0])): 
                vars_save[j].append(sol.y[j][k])
                
        variables = reflection(sol.y[:, -1], Restitution)

        #It needs to take a small step away from the edge, otherwise the solve functions terminates imediatelly  
        variables[0] = variables[0]*(1-0.0001)/np.sqrt(variables[0]**2+variables[1]**2)
        variables[1] = variables[1]*(1-0.0001)/np.sqrt(variables[0]**2+variables[1]**2) 

    return vars_save, t_save

def EM_derivs(t, vars, init_cond): #New derivatives for the electromagnetic field. It's just the Lorentz force (and velocities)
    dvdt = init_cond[0]/init_cond[1]*(init_cond[2] + np.cross(vars[3:6], init_cond[3]))
    return vars[3], vars[4], vars[5], dvdt[0], dvdt[1], dvdt[2]

def x_max(t, vars, init_cond): #Event function for solve_ivp, it stops the integration when the particle is at x=10
    return vars[0]-10

def EM_Motion(vars, dt, Tmax, init_cond, DerivFunction): #Fairly simple trajectory function, it just needs to run until it stops
    sol = solve_ivp(DerivFunction, (0, Tmax), vars, events = x_max, 
                    max_step = dt, args = (init_cond,))
    return sol.y, sol.t

#Derivatives for the gaussian field, here i take different initial conditions, 
# where the E-field has been replaced with [x0, y0, sigma]
def GaussDerivFunc(t, vars, init_cond): 
    E = np.array([-2*(vars[0]-init_cond[2][0])*np.exp(-((vars[0]-init_cond[2][0])**2+(vars[1]-init_cond[2][1])**2)/init_cond[2][2]), 
                  -2*(vars[1]-init_cond[2][1])*np.exp(-((vars[0]-init_cond[2][0])**2+(vars[1]-init_cond[2][1])**2)/init_cond[2][2]), 
                  0])
    dvdt = init_cond[0]/init_cond[1]*(E + np.cross(vars[3:6], init_cond[3]))
    return vars[3], vars[4], vars[5], dvdt[0], dvdt[1], dvdt[2]
