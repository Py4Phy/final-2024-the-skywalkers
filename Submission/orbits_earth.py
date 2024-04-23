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
au =  1.495978e8 # km (!!)

def AU(x): return x/au
def AU_inv(x): return x*au


# masses
# SI
mass_sun = 1.9885E+30 # kg
mass_earth = 5.972E+24 # kg
mass_moon = 7.35E+22 # kg
mass_endor = 7.52E+23 # kg
# mass_bb =


# Astronomical units
# DS_A: Death Star (light estimation)
# DS_B: Death Star (heavy estimation)
mass = {'Sun': 1.,
        'Earth': mass_earth/mass_sun,
        'Moon': mass_moon/mass_sun,
        'Endor': mass_endor/mass_sun,
        'DS_A': 1e18/mass_sun,
        'DS_B': 2.8e23/mass_sun
        }

# orbital period in Earth days
period = {'Earth': 365.256,
          'Moon': 1
}

# for earth: use pre-determined speed at perihelion: 30.29 km/s
speed_earth = 30.29 * 3600*24 / au


# distance from the respective orbit center in AU
# note: Earth distance at perihelion
distance = {'Earth': 1.47098074e8 / au,
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

# print(initial_velocity_earth)



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
# r: input array [[x_Earth,y_Earth],[x_Moon,y_Moon],[x_DS,y_DS]]
def F_total(r, m_earth=mass['Earth'], m_DS=mass['DS_A'], moon=False, DS=False):
    
    m_sun = mass['Sun']
    m_moon = mass['Moon']
    
    r_earth = r[0]
    r_moon = r[1]
    r_DS = r[2]
    
    # note: F_x_y means force points from y to x
    
    if moon:
        # simulate moon
        F_sun_moon = F_gravity(r_moon, m_moon, m_sun)
        F_earth_moon = F_gravity(r_earth-r_moon, m_moon, m_earth)
        
        if DS:
            # simulate DS
            F_sun_DS = F_gravity(r_DS, m_DS, m_sun)
            F_earth_DS = F_gravity(r_earth-r_DS, m_DS, m_earth)
            F_moon_DS = F_gravity(r_moon-r_DS, m_DS, m_moon)
        else:
            # no interactions with DS
            F_sun_DS = F_earth_DS = F_moon_DS = np.zeros(2)
    
    else:
        #no interactiosn with moon
        F_sun_moon = F_earth_moon = F_moon_DS = np.zeros(2)
        
        if DS:
            # simulate DS
            F_sun_DS = F_gravity(r_DS, m_DS, m_sun)
            F_earth_DS = F_gravity(r_earth-r_DS, m_DS, m_earth)
        else:
            # no interactions with DS
            F_sun_DS = F_earth_DS = np.zeros(2)
            
    # always: Earth-Sun
    F_sun_earth = F_gravity(r_earth, m_earth, m_sun)
    
    # total force
    F_tot = np.array([F_sun_earth - F_earth_moon - F_earth_DS,
                      F_sun_moon + F_earth_moon - F_moon_DS,
                      F_sun_DS + F_earth_DS + F_moon_DS])
    
    return F_tot
    



# main algorithm: integrate the orbits
def integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'],
                     dt=0.1, t_max=160, moon=False, DS=False):
    """Integrate equations of motion
    """
    nsteps = int(t_max/dt)
    time = dt * np.arange(nsteps)

    m_moon = mass['Moon']
    # # r array: time steps, 2 planets, 2 room coordinates
    # r = np.zeros((nsteps, 2, 2)) # [ [ [x_U_t1, y_U_t1], [x_N_t1, y_N_t1]], ...]
    # v = np.zeros_like(r)
    
    # r[0, 0, :] = initial_position(theta0['Uranus'], distance['Uranus'])
    # r[0, 1, :] = initial_position(theta0['Neptune'], distance['Neptune'])
    # v[0, 0, :] = initial_velocity(theta0['Uranus'], distance['Uranus'], period['Uranus'])
    # v[0, 1, :] = initial_velocity(theta0['Neptune'], distance['Neptune'], period['Neptune'])

    
    rt = np.zeros((nsteps, 3, 2)) # [[x_ea,y_ea],[x_mo,y_mo],[x_DS,y_DS]] for every time step
    vt = np.zeros_like(rt)
    

    rt[0,0,:] = initial_position(distance['Earth'])
    vt[0,0,:] = np.array([0, speed_earth])
    
  

    # integration verlocity verlet
    Ft = F_total(rt[0])
    for i in tqdm.tqdm(range(nsteps-1)):
        m = np.array([[m_earth, m_earth],
                     [m_moon, m_moon],
                     [m_DS, m_DS]])
        vhalf = vt[i] + 0.5 * dt * Ft / m
        rt[i+1] = rt[i] + dt * vhalf
    
        # new force
        Ft = F_total(rt[i+1])
        vt[i+1] = vhalf + 0.5 * dt * Ft / m

    return time, rt, vt

dt=1e-1
t_max=1*yr
time1, r1, v1 = integrate_orbits(m_earth=mass['Earth'],
                              dt=dt, t_max=t_max)
time2, r2, v2 = integrate_orbits(m_earth=mass['Earth']+mass['Moon'],
                              dt=dt, t_max=t_max)
time3, r3, v3 = integrate_orbits(m_earth=mass['Earth']+mass['DS_B'],
                              dt=dt, t_max=t_max)
time4, r4, v4 = integrate_orbits(m_earth=mass['Earth']+mass['DS_B']+mass['Moon'],
                              dt=dt, t_max=t_max)

plt.plot(r1[:,0,0], r1[:,0,1], label='Earth')
plt.plot(r2[:,0,0], r2[:,0,1], label='Earth + Moon')
plt.plot(r3[:,0,0], r3[:,0,1], label='Earth + DS_B')
plt.plot(r4[:,0,0], r4[:,0,1], label='Earth + Moon + DS_B')
plt.plot(0,0, marker='o', markersize=20, color='yellow')
plt.title('Earth\'s Orbit with different Objects')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend(loc=2)
plt.axis('equal')
plt.savefig('Earth_orbits_masses.png', dpi=300, bbox_inches='tight')




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