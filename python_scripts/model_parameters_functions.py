import numpy as np

# Seed for random number generation
entropy_seed = 12345

# Simulation parameters
T = 0.2  # simulation time in seconds
dt = 0.1  # timestep in ms
n = int(T * 1000 / dt)  # number of timesteps
n_initial = int(4000 / dt)  # initial timesteps
sens_duration = 100  # duration of sensory input
sens_input_iter = int(sens_duration / dt)  # iterations for sensory input
n_1 = int(n / 2)  # sensory implementation time

# Synaptic upscaling parameters
beta_ampa_intra = 1
beta_ampa_input = 0
beta_ampa_inter = 0
beta_gaba_pyr = 1
beta_gaba_inh = 1

# Number of cortical columns. This value change to two if the beta_ampa_inter > 0 in the SIMULATION.py
n_column = 1

# Neural gain parameters
sigma_p = 6.7

# Neural adaptation parameters
g_k_Na = 1.9

# Parameter Space of Cortical Network
c_m = 1.
tau_p, tau_i = 30., 30.
q_max_p, q_max_i = 30e-3, 60e-3
theta_p, theta_i = -58.5, -58.5
sigma_i = 6.
y_e, y_g = 70e-3, 58.6e-3
g_l = 1.
E_p_l, E_i_l = -66., -64.
E_k = -100.
E_ampa, E_gaba = 0., -70.
alpha_Na = 2.
tau_Na = 1.7
R_pump = 0.09
Na_eq = 9.5
Na_pump_0 = R_pump * (Na_eq ** 3 / (Na_eq ** 3 + 3375))
N_pp = 144
N_ii = 40
N_ip = 36
N_pi = 160
N_pINP = 16
N_iINP = 4
N_pP = 16
N_iP = 4
N_Pp = 16
N_Ip = 4
g_ampa = 1
g_gaba = 1
phi_n_sd = 1.2
i_c_m = 1 / c_m

# Define inverse variables for faster arithmetic calculations
i_tau_Na = 1 / tau_Na
i_tau_p, i_tau_i = 1 / tau_p, 1 / tau_i
i_sigma_p = 0.5 * np.pi / sigma_p / np.sqrt(3)
i_sigma_i = 0.5 * np.pi / sigma_i / np.sqrt(3)

# Define vector of variables for faster vectorized calculations
INTRA_CONN_EXC = np.array([N_pp, N_ip]).reshape(2, -1)
INTRA_CONN_INH = np.array([N_pi, N_ii]).reshape(2, -1)
INP_CONN_EXC = np.array([N_pINP, N_iINP]).reshape(2, -1)
INTER_CONN_EXC = np.array([N_pP, N_iP]).reshape(2, -1)
beta_gaba = np.array([beta_gaba_pyr, beta_gaba_inh]).reshape(2, -1)


# Define synaptic currents
def I_AMPA(gs, v):
    return g_ampa * gs * (v - E_ampa)

def I_GABA(gs, v):
    return g_gaba * gs * (v - E_gaba)

# Define firing rates
def Qp(v):
    return 0.5 * q_max_p * (1 + np.tanh((v - theta_p) * i_sigma_p))

def Qi(v):
    return 0.5 * q_max_i * (1 + np.tanh((v - theta_i) * i_sigma_i))

# Define cortical field equations
def Cortex_Field(yi, input_1):
    # Empty array to collect the numerical values
    y = np.empty(yi.shape, dtype=float)
    
    q_p, q_i = Qp(yi[0]), Qi(yi[1])
    na_aux = yi[10] * yi[10] * yi[10]
    
    # Calculate synaptic currents
    i_ampa = I_AMPA(beta_ampa_intra * yi[[2, 6]] + beta_ampa_input * yi[[11, 13]] + beta_ampa_inter * yi[[15, 17]], yi[:2])
    i_gaba = I_GABA(beta_gaba * yi[[4, 8]], yi[:2])
    
    # Dynamics in the membrane potential of excitatory populations
    y[0] = (-g_l * (yi[0] - E_p_l) - i_ampa[0] - i_gaba[0]) * i_tau_p - g_k_Na * 0.37 / (
                1 + (38.7 / yi[10]) ** 3.5) * (yi[0] - E_k) * i_c_m
    # Dynamics in the membrane potential of inhibitory populations
    y[1] = (-g_l * (yi[1] - E_i_l) - i_ampa[1] - i_gaba[1]) * i_tau_i
    
    # Dynamics in the excitatory and inhibitory synaptic currents due to local connections
    y[2:9:2] = yi[3:10:2]
    y[[3, 7]] = y_e * (y_e * (INTRA_CONN_EXC * q_p - yi[[2, 6]]) - 2 * yi[[3, 7]])
    y[[5, 9]] = y_g * (y_g * (INTRA_CONN_INH * q_i - yi[[4, 8]]) - 2 * yi[[5, 9]])
    
    # Dynamics in the concentration of Na for adaptation currents
    y[10] = (alpha_Na * q_p - (R_pump * (na_aux / (na_aux + 3375)) - Na_pump_0)) * i_tau_Na
    
    # Dynamics in the excitatory synaptic currents due to stimulus presentation through inter excitatory connections
    y[[11, 13]] = yi[[12, 14]]
    y[[12, 14]] = y_e * (y_e * (INP_CONN_EXC * input_1 - yi[[11, 13]]) - 2 * yi[[12, 14]])
    
    # Dynamics in the excitatory synaptic currents due to inter excitatory connections
    y[[15, 17]] = yi[[16, 18]]
    y[[16, 18]] = y_e * (y_e * (INTER_CONN_EXC * q_p[::-1] - yi[[15, 17]]) - 2 * yi[[16, 18]])
    return y

# Define Runge-Kutta 2nd order integration for cortical field
def RK2order_Cor(dt, data_cor, l, input_1):
    k1_cor = dt * Cortex_Field(data_cor, input_1)
    k2_cor = dt * Cortex_Field(data_cor + k1_cor + l, input_1)
    return data_cor + 0.5 * (k1_cor + k2_cor) + l

# Define function for one trial integration across simulation time
def ONE_TRIAL_INTEGRATION(RNG_init, RNG, ex_input, stochastic):
    data_collect = np.zeros((19, n_column, n), dtype=float)
    data_initi = np.empty((19, n_column, 2), dtype=float)
    l1 = np.zeros((19, n_column), dtype=float)
    data_initi[0, :, 0] = -10 * RNG_init[0].random(n_column) + theta_p
    data_initi[1, :, 0] = -10 * RNG_init[0].random(n_column) + theta_i
    data_initi[2:11, :, 0] = 0.01 * RNG_init[0].random((9, n_column))
    data_initi[11:19, :, 0] = 0
    for i in range(n_initial - 1):
        l1[[3, 7]] = y_e * y_e * np.sqrt(dt) * np.array([[RNG_init[k + kk * n_column].normal(0, phi_n_sd) for k in range(n_column)] for kk in range(2)]) * stochastic
        data_initi[:, :, 1] = RK2order_Cor(dt, data_initi[:, :, 0], l1, 0)
        data_initi[:, :, 0] = data_initi[:, :, 1]
    data_collect[:, :, 0] = data_initi[:, :, 0]
    for i in range(n - 1):
        l1[[3, 7]] = y_e * y_e * np.sqrt(dt) * np.array([[RNG[k + kk * n_column].normal(0, phi_n_sd) for k in range(n_column)] for kk in range(2)]) * stochastic
        if i >= n_1 and i < n_1 + sens_input_iter:
            input_1 = ex_input
        else:
            input_1 = 0
        data_collect[:, :, i + 1] = RK2order_Cor(dt, data_collect[:, :, i], l1, input_1)
    return data_collect

# Define function for trial simulation across simulation time
def TRIAL_SIMULATION(ex_input, n_trial, stochastic=True):
    RNG = [np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(0, 0, 0, n_column, int(10 * phi_n_sd), int(10 * beta_ampa_input), int(10 * beta_ampa_intra), int(100 * ex_input), k), )) for k in range(n_trial * 2 * n_column)]
    RNG = [RNG[j * n_column * 2:(j + 1) * n_column * 2] for j in range(n_trial)]
    RNG_init = [np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(0, 0, 1, n_column, int(10 * phi_n_sd), int(10 * beta_ampa_input), int(10 * beta_ampa_intra), int(100 * ex_input), k), )) for k in range(n_trial * 2 * n_column)]
    RNG_init = [RNG_init[j * n_column * 2:(j + 1) * n_column * 2] for j in range(n_trial)]
    data = np.zeros((n_trial, 19, n_column, n), dtype=float)
    for j in range(n_trial):
        data[j] = ONE_TRIAL_INTEGRATION(RNG_init[j], RNG[j], ex_input, stochastic)
    return data

# Define function to find beta_gaba to counteract the overexcitation due to synaptic upscaling and connecting two columns through excotatory connections
def FIND_BETA_GABA(beta_ampa_intra, beta_ampa_inter, fixedpoint):
    vp = fixedpoint[0]
    vi = fixedpoint[1]
    qp, qi = Qp(vp), Qi(vi)
    A = alpha_Na * qp / R_pump + (Na_eq ** 3) / (Na_eq ** 3 + 3375)
    NA = np.cbrt(A * 3375 / (1 - A))
    W = 0.37 / (1 + (38.7 / NA) ** 3.5)
    beta_gaba_ex_1 = -(g_l * (vp - E_p_l) + (beta_ampa_intra * N_pp + beta_ampa_inter * N_pP) * g_ampa * qp * (vp - E_ampa) + tau_p / c_m * g_k_Na * W * (vp - E_k)) / (
                g_gaba * N_pi * qi * (vp - E_gaba))
    beta_gaba_in_1 = -(g_l * (vi - E_i_l) + (beta_ampa_intra * N_ip + beta_ampa_inter * N_iP) * g_ampa * qp * (vi - E_ampa)) / (
                g_gaba * N_ii * qi * (vi - E_gaba))
    beta_gaba_ex_2 = -(g_l * (vp - E_p_l) + (beta_ampa_intra * N_pp + beta_ampa_inter * N_Pp) * g_ampa * qp * (vp - E_ampa) + tau_p / c_m * g_k_Na * W * (vp - E_k)) / (
                g_gaba * N_pi * qi * (vp - E_gaba))
    beta_gaba_in_2 = -(g_l * (vi - E_i_l) + (beta_ampa_intra * N_ip + beta_ampa_inter * N_Ip) * g_ampa * qp * (vi - E_ampa)) / (
                g_gaba * N_ii * qi * (vi - E_gaba))
    beta_gaba_ex_1 = beta_gaba_ex_1.max()
    beta_gaba_in_1 = beta_gaba_in_1.max()
    beta_gaba_ex_2 = beta_gaba_ex_2.max()
    beta_gaba_in_2 = beta_gaba_in_2.max()
    beta_gaba = np.array([[beta_gaba_ex_1, beta_gaba_ex_2], [beta_gaba_in_1, beta_gaba_in_2]])
    if beta_ampa_inter == 0:
        return beta_gaba[:, 0].reshape(2, -1)
    return beta_gaba

# Define function to find firing rate mean
def FIND_Vp_Vi(firing_p, firing_i):
    C = np.pi / 2 / np.sqrt(3)
    bins_p = np.arange(0, 31)
    bins_i = np.arange(0, 61)
    q_p_up_mean_index = np.argmax(np.histogram(firing_p[firing_p > 10], bins=bins_p)[0])
    q_p_up_mean = bins_p[q_p_up_mean_index]
    v_p_up_mean = np.arctanh(2 * q_p_up_mean / 1000 / q_max_p - 1) * sigma_p / C + theta_p
    q_i_up_mean_index = np.argmax(np.histogram(firing_i[firing_p > 10], bins=bins_i)[0])
    q_i_up_mean = bins_i[q_i_up_mean_index]
    v_i_up_mean = np.arctanh(2 * q_i_up_mean / 1000 / q_max_i - 1) * sigma_i / C + theta_i
    return np.array([v_p_up_mean, v_i_up_mean])
