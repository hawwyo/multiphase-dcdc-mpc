from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear, minimize, LinearConstraint

# np.seterr(all='raise')

@dataclass
class ConverterConfig:
    
    # Simulation time parrameters
    t_end = 2.5e-4                  # overall calculation time
    # t_end = 1                  # overall calculation time
    R_commut_time = 0.5e-4      # Time when the load transient occurs
    R_release_time = 2e-4       # Time when the load transient occurs
    R_commut_time2 = 1.4e-4     # Time when the load transient occurs
    R_release_time2 = 1.8e-4    # Time when the load transient occurs

    # Buck converter initial parameters
    f = 1e+6                  # control frequency
    T = 1 / f                 # control period

    switching_T = T / 3

    Us = 6.5                  # source voltage
    Ud = 0.9                    # desirable voltage 
    Ud_min = Ud - 70E-3       
    Ud_max = Ud + 90E-3       
    L = 47e-9               # single phase buck converter inductance [H]
    # L = 20e-9                # single phase buck converter inductance [H]
    C = 12000e-6                # output buck converter capacitance [F]    macimal is 12000uF
    idc0 = 200                 # initial total DC load current [A]
    idc1 = 1000                # load current after transient [A]
    dI = 2000*1e6            # derivative of load change: 2000A by 1us

    I_plaselimit = 100
    Uc_limit = Us
    
    I_plaselimit = 11000
    Uc_limit = Us
    
    Rl = 1e-8
    Rc = (20e-4) / 16
    # Rl = 0
    # Rc = 0

    num_phases = 16
    # num_phases = 4
    # num_phases = 2
    # num_phases = 1


class Converter:
    def __init__(self, config: ConverterConfig):
        self.config = config
        
        # self.Il = np.zeros(self.config.num_phases)
        # self.Uc = 0
        # self.Rload = self.config.Ud / self.config.idc0

        self.states = np.zeros( self.config.num_phases + 1 + 1 ) # currents + capacitor voltage + Rload
        self.duty_ratios = np.zeros( self.config.num_phases )
        self.offsets = np.linspace(start=0, stop=config.switching_T, num=config.num_phases, endpoint=False)

    def get_switch_states(self, t):
        t = np.full(self.config.num_phases, t) - self.offsets
        # print('t', t)
        number_of_full_cycles = t // np.full(self.config.num_phases, self.config.switching_T)
        # print('number_of_full_cycles', number_of_full_cycles)
        t = t - number_of_full_cycles * (self.config.switching_T)
        # print('t', t)

        return t < self.duty_ratios * (self.config.switching_T)

def calc_duty_ratios_old(states, output_voltage, config: ConverterConfig):
    Il_sum = sum(states[:-2])
    Uc = states[-2]
    Rload = states[-1]

    error = config.Ud - output_voltage
    # print('error', error)
    
    Uc_t_plus_1 = Uc + config.T * (Il_sum - Uc / Rload) / config.C
    
    Il_t_plus_1_sum_desired = config.Ud - Uc_t_plus_1
    Il_t_plus_1_sum_desired *= config.C / config.T
    Il_t_plus_1_sum_desired += Uc_t_plus_1 / Rload
    print(Il_t_plus_1_sum_desired)

    Il_t_plus_1_desired = Il_t_plus_1_sum_desired / config.num_phases
    if Il_t_plus_1_desired > 40.0:
        Il_t_plus_1_desired = 40.0

    duty_ratios = np.zeros(config.num_phases)

    for phase in range(config.num_phases):
        d = Il_t_plus_1_desired - states[phase]
        d *= config.L / config.T
        d += states[phase] * config.Rl
        d += Uc
        d /= config.Us

        if d < 0:
            d = 0
        if d > 1:
            d = 1

        duty_ratios[phase] = d
    
    return duty_ratios

cnt = 0
def calc_duty_ratios(states, output_voltage, config: ConverterConfig):
    Il = states[:config.num_phases]
    Uc = states[-2]
    Rload = states[-1]

    prediction_horizon = 5

    A = np.zeros((config.num_phases + 1, config.num_phases + 1))
    for i in range(config.num_phases):
        A[i, i] = -config.Rl / config.L
    A[-1, :] = 1 / config.C
    A[:, -1] = -1 / config.L
    A[-1, -1] = -1 / (config.C * Rload)
    A = np.identity(config.num_phases + 1) + A * config.T

    print(A)

    
    x_t = np.zeros((config.num_phases + 1, 1))
    x_t[:-1, 0] = Il
    x_t[-1, 0] = Uc



    B = np.zeros((config.num_phases + 1, config.num_phases))
    for i in range(config.num_phases):
        B[i, i] = config.Us / config.L
    B = B * config.T


    C = np.full((1, config.num_phases + 1), config.Rc)
    C[0, -1] = (Rload - config.Rc) / Rload


    n, m = C.shape
    C_diag = np.zeros(( n * prediction_horizon, m * prediction_horizon ), dtype=np.float64)
    for i in range(prediction_horizon):
        C_diag[i * n : (i * n + n),
               i * m : (i * m + m)] = C
        

    n, m = A.shape
    A_pow = np.zeros((n * prediction_horizon, m), dtype=np.float64)
    for i in range(prediction_horizon):
        A_pow[i * n : i * n + n,
              0     : m        ] = np.linalg.matrix_power(A, i + 1)
    
    D = C_diag @ A_pow @ x_t

    n, m = B.shape
    AB_pow = np.zeros((n * prediction_horizon, m * prediction_horizon), dtype=np.float64)
    for i in range(prediction_horizon):
        for j in range(i + 1):
            AB_pow[i * n : i * n + n,
                   j * m : j * m + m] = np.linalg.matrix_power(A, i - j) @ B
    
    E = C_diag @ AB_pow

    Y = np.full(D.shape[0], config.Ud, dtype=np.float64)

    def objective(x):
        diff = np.dot(E, x) + np.ravel(D) - Y
        result = np.dot(diff, diff) # + 0.001 * np.dot(x, x)

        next_state = np.ravel(A_pow @ x_t) + np.dot(AB_pow, x)
        next_il = next_state[:config.num_phases]
        std = next_il.std()
        
        std *= 1 / (config.Us ** 4)

        return result
        return result + std
    
    X_max = np.full(( prediction_horizon * (config.num_phases + 1), 1 ), config.I_plaselimit, dtype=np.float64)
    X_max[-1] = config.Uc_limit
    ub = np.ravel(X_max - A_pow @ x_t)
    lb = np.ravel( np.zeros_like(X_max) - A_pow @ x_t )
    xstate_constraint = LinearConstraint(AB_pow, ub=ub)


    # def constraint(x):
    #     pred_states = np.ravel(A @ x_t) + np.dot(AB_pow, x)
    #     return pred_states
    
    x0 = np.full(E.shape[1], config.Ud / config.Us)
    # return x0
    bounds = tuple((0, 1) for i in range(x0.size))
    # constraints = {'type': 'ineq', 'fun': constraint}
    X = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=xstate_constraint)
    # print(X)
    
    # X = lsq_linear(E, Y - np.ravel(D), (0, 1))
    print(X.x)

    global cnt
    cnt += 1

    X = X.x[:config.num_phases]

    # avg_D = sum(X) / config.num_phases
    # avg_I = sum(Il) / config.num_phases
    # if avg_I > 1e-4:
    #     for phase in range(config.num_phases):
    #         diff = (Il[phase] - avg_I) / avg_I
    #         X[phase] = avg_D - avg_D * diff

    # Without any control:
    # return np.full(config.num_phases, config.Ud / config.Us, dtype=np.float64)

    return X


def calc_output_voltage(states, config:ConverterConfig):
    
    sum_Il = sum(states[:-2])
    Uc = states[-2]
    Rload = states[-1]

    return Uc + config.Rc * (sum_Il - Uc / Rload)
    

def calc_derivatives(t, y, converter: Converter):
    Il = y[:-2]
    Uc = y[-2]
    Rload = y[-1]

    

    Uo = calc_output_voltage(y, converter.config)
    switch_states = converter.get_switch_states(t).astype(int)
    sum_Il = sum(Il)

    # print(t, switch_states, sum_Il, converter.duty_ratios)

    d_Il = (switch_states * converter.config.Us - Il * converter.config.Rl - Uc) / converter.config.L
    d_Uc = (sum_Il - Uc / Rload) / converter.config.C
    

    d_Rload = 0
    if t >= converter.config.R_commut_time:
        if t >= converter.config.R_release_time:
            if Rload < converter.config.Ud / converter.config.idc0:
                d_Rload = converter.config.dI
        else:
            if Rload > converter.config.Ud / converter.config.idc1:
                d_Rload = -converter.config.dI
            

    return np.concatenate((d_Il, np.array([d_Uc]), np.array([d_Rload])))


    




def Simulate(converter: Converter, start_time, end_time, dt, y0, steps_between_control):
    """
    Runge-Kutta 4th Order Method (ode4)
    """

    y = np.copy(y0)
    converter.states = y

    all_t = [start_time]
    all_y = [np.copy(y0)]
    all_Uo = [calc_output_voltage(y0, converter.config)]

    step = 0
    t = start_time
    while t < end_time:
        if step % steps_between_control == 0:
            print(step)
            duty_ratios = calc_duty_ratios(y, calc_output_voltage(y, converter.config), config)
            print(duty_ratios)
            print()
            converter.duty_ratios = duty_ratios
            # break

        k1 = calc_derivatives(t, y, converter)
        k2 = calc_derivatives(t + dt / 2, y + dt * k1 / 2, converter)
        k3 = calc_derivatives(t + dt / 2, y + dt * k2 / 2, converter)
        k4 = calc_derivatives(t + dt - dt, y + dt * k3, converter)

        y1 = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        y = y + y1
        y[y < 0] = 0

        # if step >= 32999 and step <= 34003:
        #     print('Y', y)

        if y[-1] < converter.config.Ud / converter.config.idc1:
            y[-1] = converter.config.Ud / converter.config.idc1
        
        if y[-1] > converter.config.Ud / converter.config.idc0:
            y[-1] = converter.config.Ud / converter.config.idc0

        t += dt

        all_t.append(t)
        all_y.append( np.copy(y) )
        all_Uo.append( calc_output_voltage(y, converter.config) )

        step += 1

        # if step == 50500:
        #     break
    
    return all_t, all_y, all_Uo


def prepare_data_for_plotting(all_converter_states):
    Il = []
    Uc = []
    total_Il = []
    for i in range(config.num_phases):
        Il.append([])
    
    for state in all_converter_states:
        total_Il.append(sum(state[:config.num_phases]))
        
        Uc.append(state[-2])
        for phase in range(config.num_phases):
            Il[phase].append( state[phase] )
    
    return Il, Uc, total_Il


def plot_min_max(axis, x, y):
    mx_y = max(y)
    mn_y = min(y)

    axis.axhline(mx_y, color='red', linestyle='--', label=f'Max = {mx_y:.2f}')
    axis.axhline(mn_y, color='green', linestyle='--', label=f'Min = {mn_y:.2f}')
    axis.legend(loc='upper left', draggable=True)

def plot_Uc(axis, x, y):    
    plot_min_max(axis, x, y)
    axis.plot(x, y)
    axis.set_xlabel('Time, seconds')
    axis.set_ylabel('Uc, Volts')
    axis.set_title('Uc')

def plot_phase_currents(axis, x, Il):
    import itertools
    plot_min_max(axis, x, list(itertools.chain.from_iterable(Il)))
    for phase in range(config.num_phases):
        axis.plot(x, Il[phase], label=f'Phase {phase}')
    axis.set_xlabel('Time, seconds')
    axis.set_ylabel('Ij, Amps') 
    axis.set_title('Phase currents')

def plot_total_current(axis, x, y):
    plot_min_max(axis, x, y)
    axis.plot(x, y)
    axis.set_xlabel('Time, seconds')
    axis.set_ylabel('I, Amps')
    axis.set_title('Total current')

def plot_output_voltage(axis, x, y):
    plot_min_max(axis, x, y)
    axis.plot(x, y)
    axis.set_xlabel('Time, seconds')
    axis.set_ylabel('Uo, Volts')
    axis.set_title('Output voltage')


if __name__ == '__main__':

    config = ConverterConfig()
    converter = Converter(config)

    # config.t_end = 0.6e-4
    D = (config.Ud / config.Us)
    shift = config.switching_T / 2

    converter.duty_ratios = np.full(config.num_phases, D)

    start_time = 0
    end_time = config.t_end

    steps_between_control = 500
    dt = config.T / steps_between_control


    initial_states = np.full( config.num_phases + 2, config.idc0 / config.num_phases)
    # initial_states = np.full( config.num_phases + 2, 0.0 )
    # initial_states[:-2] = initial_i
    initial_states[-2] = config.Ud
    initial_states[-1] = config.Ud / config.idc0

    print(initial_states)
    # initial_states = np.array([config.idc0, config.Ud, config.Ud / config.idc0])

    all_t, all_states, all_Uo = Simulate(converter=converter,
                                        start_time=start_time,
                                        end_time=end_time,
                                        dt=dt,
                                        y0=initial_states,
                                        steps_between_control=steps_between_control)
    

    Il, Uc, total_Il = prepare_data_for_plotting(all_states)

    fig, axs = plt.subplots(4, 1)

    
    plot_phase_currents(axs[0], all_t, Il)
    plot_Uc(axs[1], all_t, Uc)
    plot_total_current(axs[2], all_t, total_Il)
    plot_output_voltage(axs[3], all_t, all_Uo)   
    

    fig.set_figheight(9)
    fig.set_figwidth(8)

    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 1.2   # the amount of height reserved for white space between subplots

    # fig.subplots_adjust(left=left, 
    #                     right=right,
    #                     bottom=bottom,
    #                     top=top,
    #                     wspace=wspace,
    #                     hspace=hspace)


    fig.tight_layout()
    
    plt.show()