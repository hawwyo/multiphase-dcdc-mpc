from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

@dataclass
class ConverterConfig:
    
    # Simulation time parrameters
    t_end = 2e-4                  # overall calculation time
    # t_end = 1                  # overall calculation time
    R_commut_time = 0.5e-4      # Time when the load transient occurs
    R_release_time = 1e-4       # Time when the load transient occurs
    R_commut_time2 = 1.4e-4     # Time when the load transient occurs
    R_release_time2 = 1.8e-4    # Time when the load transient occurs

    # Buck converter initial parameters
    f = 1e+6                  # control frequency
    T = 1 / f # control period

    switching_T = T / 3

    Nconv = 16                # Number of converters
    Us = 6.5                  # source voltage
    Ud = 0.9                    # desirable voltage 
    Ud_min = Ud - 70E-3       
    Ud_max = Ud + 90E-3       
    L = 47e-9                # single phase buck converter inductance [H]
    # L = 20e-9                # single phase buck converter inductance [H]
    C = 12000e-6                # output buck converter capacitance [F]    macimal is 12000uF
    idc0 = 40                 # initial total DC load current [A]
    idc1 = 635                # load current after transient [A]
    dI = 2000*1e6             # derivative of load change: 1000A by 1us
    
    Rl = 1e-8
    Rc = 20e-4 / 16
    Rl = 0
    Rc = 0

    num_phases = 16
    num_phases = 4
    num_phases = 2
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

def calc_duty_ratios(states, output_voltage, config: ConverterConfig):
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

    d_Il = (switch_states * converter.config.Us - Il * converter.config.Rl - Uo) / converter.config.L
    d_Uc = (sum_Il - Uc / Rload) / converter.config.C
    d_Rload = 0

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
            duty_ratios = calc_duty_ratios(y, calc_output_voltage(y, converter.config), config)
            print(step, duty_ratios)
            converter.duty_ratios = duty_ratios
            # break

        k1 = calc_derivatives(t, y, converter)
        k2 = calc_derivatives(t + dt / 2, y + dt * k1 / 2, converter)
        k3 = calc_derivatives(t + dt / 2, y + dt * k2 / 2, converter)
        k4 = calc_derivatives(t + dt - dt, y + dt * k3, converter)

        y1 = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        y = y + y1
        y[y < 0] = 0

        t += dt

        all_t.append(t)
        all_y.append( np.copy(y) )
        all_Uo.append( calc_output_voltage(y, converter.config) )

        step += 1

        if step == 50100:
            break
    
    return all_t, all_y, all_Uo




if __name__ == '__main__':

    config = ConverterConfig()
    converter = Converter(config)

    config.t_end = 4e-4
    converter.duty_ratios = np.full(config.num_phases, (config.Ud / config.Us))

    start_time = 0
    end_time = config.t_end

    steps_between_control = 1000
    dt = config.T / steps_between_control

    initial_i = np.full( config.num_phases, config.idc0 / config.num_phases )
    # initial_i += 

    initial_states = np.full( config.num_phases + 2, config.idc0 / config.num_phases)
    # initial_states = np.full( config.num_phases + 2, 0.0 )
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
    

    fig, axs = plt.subplots(4, 1)

    Il = []
    Uc = []
    total_Il = []
    for i in range(config.num_phases):
        Il.append([])
    
    for state in all_states:
        total_Il.append(sum(state[:config.num_phases]))
        Uc.append(state[-2])
        for phase in range(config.num_phases):
            Il[phase].append( state[phase] )

    subset = 6100
    
    axs[0].plot(all_t, total_Il)
    axs[1].plot(all_t, Uc)
    

    for phase in range(config.num_phases):
        # axs[2].plot(all_t[:4100], Il[phase][:4100], label=f'Phase {phase}')
        axs[2].plot(all_t, Il[phase], label=f'Phase {phase}')
    
    # for x in converter.offsets:
    #     axs[2].plot([x, x], [0, 20])
    
    axs[2].legend()
    

    # axs[3].plot([all_t[-3100], all_t[-1]], [config.Ud_max, config.Ud_max])
    # axs[3].plot([all_t[-3100], all_t[-1]], [config.Ud_min, config.Ud_min])
    axs[3].plot(all_t, all_Uo)
    
    plt.show()