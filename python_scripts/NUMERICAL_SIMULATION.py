import model_parameters_functions as MPF
import numpy as np
import pandas as pd
import pickle
import os

# Define the directory
DIR = os.environ.get("DIRECTORY") + "/data/"

# Set model parameters from environment variables
MPF.beta_ampa_intra = float(os.environ.get("Beta_Intra"))
MPF.beta_ampa_input = float(os.environ.get("Beta_Input"))
MPF.beta_ampa_inter = float(os.environ.get("Beta_Inter"))
MPF.phi_n_sd = float(os.environ.get("STD_NOISE"))

# Determine if the simulation is stochastic or deterministic
stochastic = int(os.environ.get("STOCHASTIC")) == 1

# Determine if the simulation includes spontaneous activity
spontaneous = int(os.environ.get("SPONTANEOUS")) == 1

# Adjust model parameters for spontaneous activity
if spontaneous:
    MPF.T = 4
    MPF.n = int(MPF.T * 1000 / MPF.dt)

# Adjust model parameters for inter-column connections. To model one cortical column set MPF.beta_ampa_inter = 0
if MPF.beta_ampa_inter != 0:
    MPF.n_column = 2

# Set connectivity matrices based on model parameters pertaining to one or two cortical column model
MPF.INP_CONN_EXC = np.array([[MPF.N_pINP, 0], [MPF.N_iINP, 0]])
MPF.INTER_CONN_EXC = np.array([[MPF.N_pP, MPF.N_Pp], [MPF.N_iP, MPF.N_Ip]])
if MPF.beta_ampa_inter == 0:
    MPF.INP_CONN_EXC = np.array([MPF.N_pINP, MPF.N_iINP]).reshape(2, -1)
    MPF.INTER_CONN_EXC = np.array([MPF.N_pP, MPF.N_iP]).reshape(2, -1)

# Calculate beta_gaba based on model parameters to counteract overexcitation
if MPF.n_column == 2 and MPF.beta_ampa_inter == 1:
    v_fixed_point = np.load(DIR + "V_NREM.npy")
    MPF.beta_gaba = MPF.FIND_BETA_GABA(MPF.beta_ampa_intra, MPF.beta_ampa_inter, v_fixed_point)
if MPF.beta_ampa_intra != 1 or MPF.beta_ampa_inter >= 1:
    v_fixed_point = np.load(DIR + "V_WAKE.npy")
    MPF.beta_gaba = MPF.FIND_BETA_GABA(MPF.beta_ampa_intra, MPF.beta_ampa_inter, v_fixed_point)

# Set number of trials
n_trial = 1
if stochastic:
    n_trial = 500

# Determine external input
ex_input = int(os.environ.get("SLURM_ARRAY_TASK_ID")) / 100

# Perform trial simulation
data = MPF.TRIAL_SIMULATION(ex_input, n_trial, stochastic)

# Organize and save simulation data
if spontaneous:
    # Save the time trace of firing rate signals across trials
    if MPF.n_column == 2:
        data_to_save = {
            "pert": {
                "pyr": 1000 * MPF.Qp(data[:, 0, 0]),
                "inh": 1000 * MPF.Qi(data[:, 1, 0])
            },
            "unpert": {
                "pyr": 1000 * MPF.Qp(data[:, 0, 1]),
                "inh": 1000 * MPF.Qi(data[:, 1, 1])
            }
        }
    else:
        data_to_save = {
            "pert": {
                "pyr": 1000 * MPF.Qp(data[:, 0, 0]),
                "inh": 1000 * MPF.Qi(data[:, 1, 0])
            }
        }
else:
    # Save the value of firing rate signals at the stimulus offset across trials
    if MPF.n_column == 2:
        data_to_save = {
            "pert": {
                "pyr": {
                    "pre": 1000 * MPF.Qp(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 0]),
                    "post": 1000 * MPF.Qp(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 1])
                },
                "inh": {
                    "pre": 1000 * MPF.Qi(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 4]),
                    "post": 1000 * MPF.Qi(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 5])
                }
            },
            "unpert": {
                "pyr": {
                    "pre": 1000 * MPF.Qp(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 2]),
                    "post": 1000 * MPF.Qp(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 3])
                },
                "inh": {
                    "pre": 1000 * MPF.Qi(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 6]),
                    "post": 1000 * MPF.Qi(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 7])
                }
            }
        }
    else:
        data_to_save = {
            "pert": {
                "pyr": {
                    "pre": 1000 * MPF.Qp(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 0]),
                    "post": 1000 * MPF.Qp(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 1])
                },
                "inh": {
                    "pre": 1000 * MPF.Qi(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 2]),
                    "post": 1000 * MPF.Qi(data[:, :, :, [0, -1]][:, [0, 1]].reshape((n_trial, -1))[:, 3])
                }
            }
        }

# Convert data to DataFrame and save as pickle file
df = pd.DataFrame(data_to_save)
df.to_pickle(DIR+"data_stochastic_firing_{}_nColumn_{}_bIntra_{}_bInput_{}_stdnoise_{}_input_{}.pkl".format(stochastic,MPF.n_column,MPF.beta_ampa_intra,MPF.beta_ampa_input,MPF.phi_n_sd,ex_input))

# Calculate steady-state values for membrane potential of excitatory and inhibitory populations to counteract overexcitation due to connecting another cortical column or synaptic upscaling in wakefulness
if spontaneous and MPF.n_column == 1 and MPF.beta_ampa_intra == 1 and MPF.phi_n_sd == 1.2 and stochastic:
    # The expected value of membrane potential during Up state in NREM sleep is used as the steady-state value of membrane potential during wakefulness
    v_up = MPF.FIND_Vp_Vi(1000 * MPF.Qp(data[:, 0, 0]), 1000 * MPF.Qi(data[:, 1, 0]))
    np.save(DIR + "V_WAKE.npy", v_up)
    
    # The steady-state value for the one-cortical column model in NREM sleep is used as the steady-state value for the two-cortical column model in NREM sleep
    data_steady = MPF.TRIAL_SIMULATION(0., 1, stochastic=False)
    np.save(DIR + "V_NREM.npy", data_steady[0, :2, 0, -1])
