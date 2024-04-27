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
# import integrators



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
mass_earth = 5.9724E+24 # kg
mass_moon = 7.346E+22 # kg
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
speed_earth = 30.29 * 3600*24 / au # speed earth around sun at perihelion
# for moon: speed at apogee: 0.966 km/s
speed_moon =  0.966 * 3600*24 / au # speed moon around earth at apogee
# for DS: number that lead to stable orbits
speed_DS = 3.99 * 3600*24 / au # speed DS around earth


# distance from the respective orbit center in AU
# note: Earth distance at perihelion (closest), Moon at apogee (farthest)
distance = {'Earth': 1.47098074e8 / au, # e8 is in km!!
            'Moon': 4.055e5 / au,
            'DS': 2.5e4 / au,
            }

# radii of the objects
radius = {'Sun':696000 / au,
          'Earth': 6357 / au,
          'Moon': 1737 / au,
          'Endor': 2450 / au,
          'BB': 74000 / au,
          'DS': 100 / au
          }


# =============================================================================
# 2. Functions for the orbit calculations (Solar System) (for Endor System
# see other file in Submission)
# =============================================================================


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


def dist(r1, r2):
    '''
    Parameters r1, r2: np.array of shape r = [x,y]
    Returns absolute value of the distance of connection vector
    '''
    return np.sqrt(np.sum((r1-r2)**2))


def crash_detect(r):
    crash = False
    [r1, r2, r3] = r
    if dist(r1, [0,0]) < (radius['Sun']+radius['Earth']):
        print('\nEarth crashed into the Sun!')
        crash = True
    elif dist(r1, r2) < (radius['Moon']+radius['Earth']):
        print('\nMoon crashed into the Earth!')
        crash = True
    elif dist(r1, r3) < (radius['DS']+radius['Earth']):
        print('\nDeathstar crashed into the Earth!')
        crash = True
    elif dist(r3, np.array([0,0])) < (radius['DS']+radius['Sun']):
        print('\nDeathstar crashed into the Sun!')
        crash = True
        
    return crash


# calculate the total force
# r: input array [[x_Earth,y_Earth],[x_Moon,y_Moon],[x_DS,y_DS]]
def F_total(r, m_earth=mass['Earth'], m_DS=mass['DS_A'],
            sun=True, moon=False, DS=False, earth_fixed=False):
    
    # masses
    m_sun = mass['Sun']
    m_moon = mass['Moon']
    
    # radii of objects
    r_earth = r[0]
    r_moon = r[1]
    r_DS = r[2]
    
    # deactivate sun if only orbits around Earth shall be observed
    sunfactor = (1 if sun else 0) # ;)
    
    # deactivate forces acting on earth if earth should be held fixed
    earthfactor = (0 if earth_fixed else 1)
    
    # note: F_x_y means force points from y to x
    if moon:
        # simulate moon
        F_sun_moon = F_gravity(r_moon, m_moon, m_sun) * sunfactor
        F_earth_moon = F_gravity(r_moon-r_earth, m_moon, m_earth)
        
        if DS:
            # simulate DS
            F_sun_DS = F_gravity(r_DS, m_DS, m_sun) * sunfactor
            F_earth_DS = F_gravity(r_DS-r_earth, m_DS, m_earth)
            F_moon_DS = F_gravity(r_DS-r_moon, m_DS, m_moon)
        else:
            # no interactions with DS
            F_sun_DS = F_earth_DS = F_moon_DS = np.zeros(2)
    
    else:
        #no interactiosn with moon
        F_sun_moon = F_earth_moon = F_moon_DS = np.zeros(2)
        
        if DS:
            # simulate DS
            F_sun_DS = F_gravity(r_DS, m_DS, m_sun) * sunfactor
            F_earth_DS = F_gravity(r_DS-r_earth, m_DS, m_earth)
        else:
            # no interactions with DS
            F_sun_DS = F_earth_DS = np.zeros(2)
            
    # always: Earth-Sun
    F_sun_earth = F_gravity(r_earth, m_earth, m_sun) * sunfactor
    
    # total force
    F_tot = np.array([F_sun_earth - F_earth_moon - F_earth_DS,
                      F_sun_moon + F_earth_moon - F_moon_DS,
                      F_sun_DS + F_earth_DS + F_moon_DS])
    # account for fixed earht (=setting all forces acting on earth to zero)
    F_tot[0] *= earthfactor
    
    return F_tot
    


# main algorithm: integrate the orbits
def integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'], 
                     distance_DS=distance['DS'], _speed_DS=speed_DS,
                     dt=0.1, t_max=160, sun=True, moon=False, DS=False,
                     earth_fixed=False):
    """Integrate equations of motion
    """
    nsteps = int(t_max/dt)
    time = dt * np.arange(nsteps)
    
    # masses
    m_moon = mass['Moon']
    
    # speeds from global variables
    _speed_earth = speed_earth * (1 if sun else 0) # when sun neglected, see earth as center
    _speed_moon = speed_moon
    
    rt = np.zeros((nsteps, 3, 2)) # [[x_ea,y_ea],[x_mo,y_mo],[x_DS,y_DS]] for every time step
    vt = np.zeros_like(rt)
    
    # initialising earth
    rt[0,0,:] = initial_position(distance['Earth'])
    vt[0,0,:] = np.array([0, _speed_earth])
    
    # initialising moon
    rt[0,1,:] = initial_position(distance['Earth']) + np.array([0,distance['Moon']])
    vt[0,1,:] = np.array([- _speed_moon, _speed_earth])
    
    # initialising Death Star
    rt[0,2,:] = initial_position(distance['Earth']) + np.array([0,-distance_DS])
    vt[0,2,:] = np.array([1*_speed_DS, _speed_earth])
    
    # print(np.sqrt(np.sum((vt[0])**2, axis=1)))
    # print()

    # integration verlocity verlet
    Ft = F_total(rt[0], m_earth=m_earth, m_DS=m_DS, sun=sun, moon=moon, DS=DS,
                 earth_fixed=earth_fixed)
    
    for i in tqdm.tqdm(range(nsteps-1)):
        # print(vt[i,0])
        # print(dist(rt[i,2], np.array([0,0])), Ft[2])
        m = np.array([[m_earth, m_earth],
                     [m_moon, m_moon],
                     [m_DS, m_DS]])
        vhalf = vt[i] + 0.5 * dt * Ft / m
        rt[i+1] = rt[i] + dt * vhalf
    
        # new force
        Ft = F_total(rt[i+1], m_earth=m_earth, m_DS=m_DS, moon=moon, DS=DS,
                     sun=sun, earth_fixed=earth_fixed)
        # print(Ft[1])
        vt[i+1] = vhalf + 0.5 * dt * Ft / m
        # print(np.sqrt(np.sum((vt[i+1])**2, axis=1)))
        
        # crash detection: important to stop, otherwise F explodes
        if crash_detect(rt[i+1]):
            print('STOP SIMULATION')
            print('CELESTIAL CATASTROPHE, PEOPLE DIED!')
            # set all remaining r to the current r[i]
            rt = np.where(rt==np.zeros((3,2)), rt[i+1], rt)
            break

    return time, rt, vt


def orbit_time(r, t, neglect_first=100, eps=1e-4):
    r0 = r[0]
    for index, ri in enumerate(r[neglect_first:]):
        if dist(r0, ri) < eps:
            print('Orbit time successully calculated.')
            return time[neglect_first + index]
    else:
        print('Orbit time could NOT be calculated.')
    return None


def stretch_distance(rt, index=1, alpha=30):
    r1 = rt[:,0]
    r2 = rt[:,index]
    rt[:,index] = r1 + alpha * (r2 - r1)
    return rt, '(stretched)'
        

dt=1e-3
t_max=3
sun=False
DS=True
moon=False
earth_fixed=True
moon_stretch = ''
DS_stretch = ''

time, r, v = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'],
                                 sun=sun, moon=moon, DS=DS, dt=dt, t_max=t_max,
                                 earth_fixed=earth_fixed)


# plt.plot(0,0, marker='o', markersize=20, color='yellow')
# plt.show()
    
# r, moon_stretch = stretch_distance(r, 1)
r, DS_stretch = stretch_distance(r, 2, 10)

plt.plot(r[:,0,0], r[:,0,1], c='b')
plt.plot(r[0,0,0], r[0,0,1], marker='o', c='b', markersize=8, label='Earth')

if moon:
    plt.plot(r[:,1,0], r[:,1,1], c='grey')
    plt.plot(r[0,1,0], r[0,1,1], marker='o', c='grey', markersize=5, 
             label=f'Moon {moon_stretch}')

if DS:
    plt.plot(r[:,2,0], r[:,2,1], c='black')
    plt.plot(r[0,2,0], r[0,2,1], marker='o', c='black', markersize=5,
             label=f'DS {DS_stretch}')
    
# plt.ylim(-2, 2)
# plt.plot(r3[:,0,0], r3[:,0,1], label='Earth + DS_B')
# plt.plot(r4[:,0,0], r4[:,0,1], label='Earth + Moon + DS_B')
# if sun:
#     plt.plot(0,0, marker='o', markersize=20, color='yellow')

plt.title('DS_A and Moon orbiting around Earth')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.legend(loc=2)
plt.axis('equal')
# plt.savefig('earth_moon_DS_A.png', dpi=300, bbox_inches='tight')
plt.show()

print(orbit_time(r[:,2], time, eps=4e-4))

# =============================================================================
# Compare omega
# =============================================================================

# t_max = yr

# dt=1e-1
# time1, r1, v1 = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_A'],
#                               moon=False, DS=False, dt=dt, t_max=t_max)

# dt=1e-1
# time2, r2, v2 = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_B'],
#                               moon=True, DS=False, dt=dt, t_max=t_max)

# dt=1e-3
# time3, r3, v3 = integrate_orbits(m_earth=mass['Earth'], m_DS=mass['DS_B'],
#                               moon=False, DS=True, dt=dt, t_max=t_max)


# o1 = omega(v1[:,0], r1[:,0])
# o2 = omega(v2[:,0], r2[:,0])
# o3 = omega(v3[:,0], r3[:,0])
# plt.plot(time3,o3, label='Earth, DS_B', c='grey')
# plt.plot(time1,o1, label='Earth')
# plt.plot(time2,o2, label='Earth, Moon')
# plt.legend()
# plt.xlabel('t [days]')
# plt.ylabel('$\omega$ [AU/day]')
# plt.title('Angular Velocity of Earth (1 year)')
# plt.savefig('omega_earth_1year.png', dpi=300, bbox_inches='tight')


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