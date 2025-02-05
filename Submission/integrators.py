# integrators

def rk4(y, f, t, h):
    """Runge-Kutta RK4"""
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)
    return y + (k1 + 2*k2 + 2*k3 + k4)*h/6


def velocity_verlet(y, f, t, h):
    """Velocity Verlet

    Low-performance implementation because the force is calculated
    twice; should remember the second force calculation and use as
    input for the next step.

    For comparing different algorithms it is ok to use this
    implementation for convenience. For production level code you
    should use a better implementation that saves the second force
    evaluation.

    """
    # half step velocity
    F = f(t, y)
    y[1] += 0.5*h * F[1]
    # full step position
    y[0] += h*y[1]
    # full step velocity (updated positions!)
    # NOTE: this force evaluation should be used for the next iteration!!!
    F = f(t+h, y)
    y[1] += 0.5*h * F[1]

    return y
