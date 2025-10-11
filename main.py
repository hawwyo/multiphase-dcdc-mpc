from dataclasses import dataclass
import numpy as np

@dataclass
class ConverterConfig:
    
    # Simulation time parrameters
    t_end = 2e-4                  # overall calculation time
    R_commut_time = 0.5e-4      # Time when the load transient occurs
    R_release_time = 1e-4       # Time when the load transient occurs
    R_commut_time2 = 1.4e-4     # Time when the load transient occurs
    R_release_time2 = 1.8e-4    # Time when the load transient occurs

    # Buck converter initial parameters
    f = 1e+6                  # control frequency
    T = 1 / f # control period

    Nconv = 16                # Number of converters
    Us = 6.5                  # source voltage
    Ud = 0.9                    # desirable voltage 
    # self.Ud_min = self.Ud - 70E-3       
    # self.Ud_max = self.Ud + 90E-3       
    L = 47e-9                # single phase buck converter inductance [H]
    C = 12000e-6                # output buck converter capacitance [F]    macimal is 12000uF
    idc0 = 20                 # initial total DC load current [A]
    idc1 = 635                # load current after transient [A]
    dI = 2000*1e6             # derivative of load change: 1000A by 1us
    
    Rl = 1e-8
    Rc = 20e-4 / 16

    num_phases = 16


class Converter:
    def __init__(self, config: ConverterConfig):
        self.config = config
        
        # self.Il = np.zeros(self.config.num_phases)
        # self.Uc = 0
        # self.Rload = self.config.Ud / self.config.idc0

        self.states = np.zeros( self.config.num_phases + 1 + 1 ) # currents + capacitor voltage + Rload
        self.duty_ratios = np.zeros( self.config.num_phases )
        self.offsets = np.linspace(start=0, stop=config.T, num=config.num_phases, endpoint=False)

    def get_switch_states(self, t):
        t = np.full(self.config.num_phases, t) - self.offsets
        number_of_full_cycles = t // np.full(self.config.num_phases, self.config.T)
        t = t - number_of_full_cycles * self.config.T

        return t <= self.duty_ratios


def calc_output_voltage(states, config:ConverterConfig):
    states = states
    sum_Il = sum(states[:-2])
    Uc = states[-2]
    Rload = states[-1]

    return Uc + config.Rc * (sum_Il - Uc / Rload)
    

def calc_derivatives(t, y, converter: Converter):
    Il = y[:-2]
    Uc = y[-2]
    Rload = y[-1]

    dy = np.zeros(y.size())

    Uo = calc_output_voltage(y, converter.config)
    switch_states = converter.get_switch_states(t)
    sum_Il = sum(Il)

    d_Il = (switch_states * converter.config.Us - Il * converter.config.Rl - Uo) / converter.config.L
    d_Uc = (sum_Il - Uo / Rload) / converter.config.C
    d_Rload = 0

    return np.concatenate((d_Il, d_Uc, d_Rload))


    




def Simulate(converter: Converter, start_time, end_time, dt, y0):
    """
    Runge-Kutta 4th Order Method (ode4)

    Parameters:
        f  : function f(t, y) that returns dy/dt
        y0 : initial value (can be a scalar or numpy array)
        t  : array of time points (e.g., numpy.linspace)

    Returns:
        y  : array of solution values at each time step
    """

    y = np.copy(y0)
    all_y = [np.copy(y0)]

    t = start_time
    while t < end_time:
        k1 = calc_derivatives(t, y, converter)
        k2 = calc_derivatives(t + dt / 2, y + dt * k1 / 2, converter)
        k3 = calc_derivatives(t + dt / 2, y + dt * k2 / 2, converter)
        k4 = calc_derivatives(t + dt, y + dt * k3, converter)

        y1 = h

    for i in range(1, len(t)):
        dt = step
        ti = t[i - 1]
        yi = y[i - 1]

        k1 = f(ti, yi)
        k2 = f(ti + dt / 2.0, yi + dt * k1 / 2.0)
        k3 = f(ti + dt / 2.0, yi + dt * k2 / 2.0)
        k4 = f(ti + dt, yi + dt * k3)

        y[i] = yi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y



if __name__ == '__main__':

    config = ConverterConfig()
    converter = Converter(config)

    converter.duty_ratios = np.full(config.num_phases, config.T / 16)

    start = config.T / 16 / 2
    for i in range(config.num_phases * 3):
        print( converter.get_switch_states(start).astype(int) )
        start += config.T / 16