# PHY432: Final Project
# Team: The Skywalkers
# Members: Simon Tebeck, Pranav Gupta
# April 2024


# import packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("ggplot")
import tqdm

# integrators
import integrators



# =============================================================================
# 1. Initialising parameters
# =============================================================================

# gravitational constant
G_gravity = 4*np.pi**2 *(1/365.256)**2 # astronomical units but per days not years
yr = 365.256 # days

# astronomical unit
au =  1.495978e11 # m

def AU(x): return x/au
def AU_inv(x): return x*au


# masses
# SI
m_sun = 1.9885E+30 # kg
m_earth = 5.972E+24 # kg
m_moon = 7.35E+22 # kg
m_endor = 7.52E+23 # kg
# m_bb =


# Astronomical units
mass = {'Sun': 1.,
        'Earth': m_earth/m_sun,
        'Moon': m_moon/m_sun,
        'Endor': m_endor/m_sun
        }

# orbital period in Earth days
period = {'Earth': 365.256,
          'Moon': 1
}


# distance from the respective orbit center in AU
distance = {'Earth': 1,
            'Moon': 3.85e8 / au,
}

# }


#------------------------------------------------------------


def initial_position(distance, angle=0):
    """Calculate initial planet position.

    Parameters
    ----------
    angle : float
       initial angle relative to x axis (in degrees)
    distance : float
       initial distane from sun (in AU)

    Returns
    -------
    array
       position (x, y)
    """
    x = np.deg2rad(angle)
    return distance * np.array([np.cos(x), np.sin(x)])


def initial_velocity(distance, period, angle=0):
    # Convert angle from degrees to radians
    angle_rad = angle * (np.pi / 180)

    velocity_au_day = 2 * np.pi * distance / period

    # Calculate x and y components of the velocity
    vx = velocity_au_day * -np.sin(angle_rad)
    vy = velocity_au_day * np.cos(angle_rad)

    return np.array([vx, vy])


# Calculate the initial velocities for Earth
key = 'Earth'
initial_velocity_earth = initial_velocity(distance[key], 
                                            period[key], angle=0)





def F_gravity(r, m, M):
    """Force due to gravity between two masses.

    Parameters
    ----------
    r : array
      distance vector (x, y)
    m, M : float
      masses of the two bodies

    Returns
    -------
    array
       force due to gravity (along r)
    """
    rr = np.sum(r * r)
    rhat = r / np.sqrt(rr)
    force_magnitude = - G_gravity * m * M / rr* rhat
    return force_magnitude


def omega(v, r):
    """Calculate angular velocity.

    The angular velocity is calculated as
    .. math::

          \omega = \frac{|\vec{v}|}{|\vec{r}|}

    Parameters
    ----------
    v : array
       velocity vectors for all N time steps; this
       should be a (N, dim) array
    r : array
       position vectors (N, dim) array

    Returns
    -------
    array
       angular velocity for each time step as 1D array of
       length N
    """
    speed = np.linalg.norm(v, axis=1)
    distance = np.linalg.norm(r, axis=1)
    return speed/distance


# calculate the total force
def F_total(r):
    m_Sun = mass['Sun']
    m_Earth = mass['Earth']
    
    F_sun = F_gravity(r, m_Earth, m_Sun)
    
    return F_sun
    
    
# def F_total(r, coupled=True):
    
#     rU = r[0]
#     rN = r[1]
#     mU = mass['Uranus']
#     mN = mass['Neptune']
#     mS = mass['Sun']
    
#     # force from Sun
#     F_sun_U = F_gravity(rU, mU, mS)
#     F_sun_N = F_gravity(rN, mN, mS)
    
#     if coupled:
#         # calculate only for Uranus, the one for Neptune is that one negative
#         F_other = F_gravity(rU - rN, mU, mN)
#     else:
#         F_other = np.array([0,0])
    
#     F_total = np.array([F_sun_U + F_other, F_sun_N - F_other])
#     return F_total


# main algorithm: integrate the orbits
def integrate_orbits(dt=0.1, t_max=160, coupled=True):
    """Integrate equations of motion
    """
    nsteps = int(t_max/dt)
    time = dt * np.arange(nsteps)

    # # r array: time steps, 2 planets, 2 room coordinates
    # r = np.zeros((nsteps, 2, 2)) # [ [ [x_U_t1, y_U_t1], [x_N_t1, y_N_t1]], ...]
    # v = np.zeros_like(r)
    
    # r[0, 0, :] = initial_position(theta0['Uranus'], distance['Uranus'])
    # r[0, 1, :] = initial_position(theta0['Neptune'], distance['Neptune'])
    # v[0, 0, :] = initial_velocity(theta0['Uranus'], distance['Uranus'], period['Uranus'])
    # v[0, 1, :] = initial_velocity(theta0['Neptune'], distance['Neptune'], period['Neptune'])

    
    r = np.zeros((nsteps, 2)) # [x1,y1],[x2,y2],...
    v = np.zeros_like(r)

    r[0,:] = initial_position(distance['Earth'])
    v[0,:] = initial_velocity(distance['Earth'], period['Earth'])
    

    m_Earth = mass['Earth']

    # integration verlocity verlet
    Ft = F_total(r[0])
    for i in range(nsteps-1):
        m = np.array([m_Earth, m_Earth])
        vhalf = v[i] + 0.5 * dt * Ft / m
        r[i+1] = r[i] + dt * vhalf
    
        # new force
        Ft = F_total(r[i+1])
        v[i+1] = vhalf + 0.5 * dt * Ft / m

    return time, r, v


time, r, v = integrate_orbits(dt=1e-1, t_max=1*yr)
plt.plot(r[:,0], r[:,1])
plt.axis('equal')




# # Simulatirint("Simulating Uranus and Neptune orbits WITHOUT interactions")
# time, r0, v0 = integrate_orbits(t_max=160, coupled=False)
# rU0 = r0[:, 0]  # Uranus position without interaction
# vU0 = v0[:, 0]  # Uranus velocity without interaction
# omegaU0 = omega(vU0, rU0)  # Angular velocity of Uranus without interaction

# # Simulating orbits WITH interactions
# print("Simulating Uranus and Neptune orbits WITH interactions")
# time, r, v = integrate_orbits(t_max=160, coupled=True)
# rU = r[:, 0]  # Uranus position with interaction
# vU = v[:, 0]  # Uranus velocity with interaction
# omegaU = omega(vU, rU)  # Angular velocity of Uranus with interaction

# # Calculate DeltaOmega for Uranus
# DeltaOmegaU = omegaU - omegaU0

# # Plotting the orbits
# plt.figure(figsize=(12, 6))
# plt.suptitle('Orbits of Uranus and Neptune around the Sun', fontsize=25)

# # Plot orbits without interactions
# plt.subplot(1, 2, 1)
# plt.plot(rU0[:, 0], rU0[:, 1], label="Uranus (No Interaction)")
# plt.plot(r0[:, 1, 0], r0[:, 1, 1], label="Neptune (No Interaction)")
# plt.title("Orbits Without Interactions")
# plt.xlabel("x (AU)")
# plt.ylabel("y (AU)")
# plt.legend()

# # Plot orbits with interactions
# plt.subplot(1, 2, 2)
# plt.plot(rU[:, 0], rU[:, 1], label="Uranus (With Interaction)")
# plt.plot(r[:, 1, 0], r[:, 1, 1], label="Neptune (With Interaction)")
# plt.title("Orbits With Interactions")
# plt.xlabel("x (AU)")
# plt.ylabel("y (AU)")
# plt.legend()

# plt.tight_layout()
# plt.savefig('uranus_neptune_orbits.png')
   
# ng orbits WITHOUT interactions
# p
# # plot of delta omega
# plt.figure()
# plt.plot(time, DeltaOmegaU)
# plt.xlabel('time [year]')
# plt.ylabel('$\Delta\omega$ [rad/year]')
# plt.title('Anomaly of $\Delta\omega$')
# plt.savefig('uranus_anomaly.png')