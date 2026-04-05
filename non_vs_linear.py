import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def nonlinear(t, y, h):
    x, v = y
    dxdt = v
    dvdt = -2*x*(1 - 1/np.sqrt((1+h)**2 + x**2))
    return [dxdt, dvdt]

def linear(t, y, h):
    x, v = y
    omega2 = 2*h/(1+h)
    return [v, -omega2*x]

t_span = (0, 30)
t_eval = np.linspace(0, 30, 1000)

for i, h in enumerate([1, 0.5, 0.1]):
    sol_nl = solve_ivp(nonlinear, t_span, [0,1], t_eval=t_eval, args=(h,))
    sol_lin = solve_ivp(linear, t_span, [0,1], t_eval=t_eval, args=(h,))

    np.savetxt(f"nonlinear_h{i + 1}.dat", np.column_stack((t_eval, sol_nl.y[0])))
    np.savetxt(f"linear_h{i + 1}.dat", np.column_stack((t_eval, sol_lin.y[0])))
    
    plt.figure()
    plt.plot(t_eval, sol_nl.y[0], label="Nonlinear")
    plt.plot(t_eval, sol_lin.y[0], "--", label="Linear")
    plt.title(f"h = {h}")
    plt.legend()
    plt.show()
