import functions_analysis as FA
import model_parameters_functions as MPF
import numpy as np
import pickle
import pandas as pd
import os
from mpi4py import MPI

# Set directory
DIR = os.environ.get("DIRECTORY") + "/data/"

# Initialize MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Function to compute deterministic respons
def Deterministic_Response(bintra, binput, binter, V_WAKE, V_NREM, ex_input):
    """
    Perform deterministic response simulation of a neural model.

    Args:
        bintra (float): Synaptic upscaling factor of intra-columnar AMPA connectivity.
        binput (float): Synaptic upscaling factor of stimulus AMPA connectivity.
        binter (float): Synaptic upscaling factor of inter-columnar AMPA connectivity.
        V_WAKE (float): Wake steady-state membrane potential.
        V_NREM (float): NREM steady-state membrane potential.
        ex_input (array): External input to the model.

    Returns:
        DataFrame: DataFrame containing simulation results.
    """

    def Obtain_E_I(data, column, j, timepoint):
        """
        Calculate excitatory and inhibitory currents for a given population and timepoint.

        Args:
            data (array): Model data containing dynamical variables.
            column (int): Column index.
            j (int): Population index.
            timepoint (int): Timepoint index.

        Returns:
            tuple: Tuple containing excitatory and inhibitory currents.
        """

        EXC_i_ampa_local = MPF.I_AMPA(MPF.beta_ampa_intra * data[0, 4 * j + 2, column, timepoint],
                                      data[0, j, column, timepoint])
        EXC_i_gaba = MPF.I_GABA(MPF.beta_gaba[j, 0] * data[:, 4 * j + 4, column, timepoint],
                                data[0, j, column, timepoint])

        EXC_i_ampa_input = MPF.I_AMPA(MPF.beta_ampa_input * data[0, 2 * j + 11, column, timepoint],
                                      data[0, j, column, timepoint])

        EXC_i_ampa_inter = 0
        if MPF.n_column == 2:
            EXC_i_ampa_inter = MPF.I_AMPA(MPF.beta_ampa_inter * data[0, 2 * j + 15, column, timepoint],
                                          data[0, j, column, timepoint])

        EXC_I = EXC_i_gaba[0]

        current_leak = MPF.g_l * (data[0, j, column, timepoint] - MPF.E_i_l)
        current_kna = 0

        if j == 0:
            current_leak = MPF.g_l * (data[0, j, column, timepoint] - MPF.E_p_l)
            current_kna = MPF.g_k_Na * 0.37 / (1 + (38.7 / data[0, 10, column, timepoint]) ** 3.5) * (
                        data[0, j, column, timepoint] - MPF.E_k) * MPF.i_c_m / MPF.i_tau_p

        return -EXC_i_ampa_local, -EXC_i_ampa_inter, EXC_I, -EXC_i_ampa_input, current_leak, current_kna

    # Set synaptic upscaling factors for the AMPAergic connectivities
    MPF.beta_ampa_intra = bintra
    MPF.beta_ampa_input = binput
    MPF.beta_ampa_inter = binter

    MPF.phi_n_sd = 1.2  # Set noise level
    stochastic = False  # Deterministic simulation

    MPF.T = 4  # Simulation time
    MPF.n = int(MPF.T * 1000 / MPF.dt)  # Number of time steps
    MPF.n_1 = 10000  # Timepoint index

    # Set up connectivity matrices
    if MPF.beta_ampa_inter != 0:
        MPF.n_column = 2

    MPF.INP_CONN_EXC = np.array([[MPF.N_pINP, 0], [MPF.N_iINP, 0]])
    MPF.INTER_CONN_EXC = np.array([[MPF.N_pP, MPF.N_Pp], [MPF.N_iP, MPF.N_Ip]])
    if MPF.beta_ampa_inter == 0:
        MPF.INP_CONN_EXC = np.array([MPF.N_pINP, MPF.N_iINP]).reshape(2, -1)
        MPF.INTER_CONN_EXC = np.array([MPF.N_pP, MPF.N_iP]).reshape(2, -1)

    # Determine strenght of GABAergic synapses to counteract the overexcitation
    if MPF.n_column == 2 and MPF.beta_ampa_inter == 1:
        v_fixed_point = V_NREM
        MPF.beta_gaba = MPF.FIND_BETA_GABA(MPF.beta_ampa_intra, MPF.beta_ampa_inter, v_fixed_point)
    if MPF.beta_ampa_intra != 1 or MPF.beta_ampa_inter >= 1:
        v_fixed_point = V_WAKE
        MPF.beta_gaba = MPF.FIND_BETA_GABA(MPF.beta_ampa_intra, MPF.beta_ampa_inter, v_fixed_point)

    n_trial = 1  # Number of trials

    # Perform trial simulation
    data = MPF.TRIAL_SIMULATION(ex_input, n_trial, stochastic)

    # Organize simulation results into DataFrame
    if MPF.n_column == 2:
        data_to_save = {
            "pert": {
                "pyr": {
                    "firing": 1000 * MPF.Qp(data[:, 0, 0]),
                    "E/I_pre": Obtain_E_I(data, 0, 0, MPF.n_1 - 1000),
                    "E/I_post": Obtain_E_I(data, 0, 0, MPF.n_1 + 1000)
                },
                "inh": {
                    "firing": 1000 * MPF.Qi(data[:, 1, 0]),
                    "E/I_pre": Obtain_E_I(data, 0, 1, MPF.n_1 - 1000),
                    "E/I_post": Obtain_E_I(data, 0, 1, MPF.n_1 + 1000)
                }
            },
            "unpert": {
                "pyr": {
                    "firing": 1000 * MPF.Qp(data[:, 0, 1]),
                    "E/I_pre": Obtain_E_I(data, 1, 0, MPF.n_1 - 1000),
                    "E/I_post": Obtain_E_I(data, 1, 0, MPF.n_1 + 1000)
                },
                "inh": {
                    "firing": 1000 * MPF.Qi(data[:, 1, 1]),
                    "E/I_pre": Obtain_E_I(data, 1, 1, MPF.n_1 - 1000),
                    "E/I_post": Obtain_E_I(data, 1, 1, MPF.n_1 + 1000)
                }
            }
        }
    else:
        data_to_save = {
            "pert": {
                "pyr": {
                    "firing": 1000 * MPF.Qp(data[:, 0, 0]),
                    "E/I_pre": Obtain_E_I(data, 0, 0, MPF.n_1 - 1000),
                    "E/I_post": Obtain_E_I(data, 0, 0, MPF.n_1 + 1000)
                },
                "inh": {
                    "firing": 1000 * MPF.Qi(data[:, 1, 0]),
                    "E/I_pre": Obtain_E_I(data, 0, 1, MPF.n_1 - 1000),
                    "E/I_post": Obtain_E_I(data, 0, 1, MPF.n_1 + 1000)
                }
            }
        }

    # Convert data to DataFrame
    df = pd.DataFrame(data_to_save)

    return df

# Function to obtain Excitatory and Inhibitory currents for EI Trace
def Obtain_EI_Trace(bintra, binput, binter, V_WAKE, ex_input):
    """
    Obtain excitatory and inhibitory trace for a given simulation setup.

    Args:
        bintra (float): Synaptic upscaling factor of intra-columnar AMPA connectivity.
        binput (float): Synaptic upscaling factor of stimulus AMPA connectivity.
        binter (float): Synaptic upscaling factor of inter-columnar AMPA connectivity.
        V_WAKE (float): Wake state membrane potential.
        ex_input (array): External input to the model.

    Returns:
        float: Excitatory and inhibitory trace.
    """

    j = 0
    column = 0

    # Set synaptic upscaling factors for the AMPAergic connectivities
    MPF.beta_ampa_intra = bintra
    MPF.beta_ampa_input = binput
    MPF.beta_ampa_inter = binter

    MPF.phi_n_sd = 1.2  # Set noise level
    stochastic = False  # Deterministic simulation

    MPF.T = 4  # Simulation time
    MPF.n = int(MPF.T * 1000 / MPF.dt)  # Number of time steps
    MPF.n_1 = 10000  # Timepoint index

    # Set up connectivity matrices
    if MPF.beta_ampa_inter != 0:
        MPF.n_column = 2

    MPF.INP_CONN_EXC = np.array([[MPF.N_pINP, 0], [MPF.N_iINP, 0]])
    MPF.INTER_CONN_EXC = np.array([[MPF.N_pP, MPF.N_Pp], [MPF.N_iP, MPF.N_Ip]])
    if MPF.beta_ampa_inter == 0:
        MPF.INP_CONN_EXC = np.array([MPF.N_pINP, MPF.N_iINP]).reshape(2, -1)
        MPF.INTER_CONN_EXC = np.array([MPF.N_pP, MPF.N_iP]).reshape(2, -1)

    # Determine strenght of GABAergic synapses to counteract the overexcitation
    v_fixed_point = V_WAKE
    MPF.beta_gaba = MPF.FIND_BETA_GABA(MPF.beta_ampa_intra, MPF.beta_ampa_inter, v_fixed_point)

    n_trial = 1  # Number of trials

    # Perform trial simulation
    data = MPF.TRIAL_SIMULATION(ex_input, n_trial, stochastic)

    # Calculate excitatory and inhibitory currents
    EXC_i_ampa_local = MPF.I_AMPA(MPF.beta_ampa_intra * data[0, 4 * j + 2, column], data[0, j, column])
    EXC_i_gaba = MPF.I_GABA(MPF.beta_gaba[j, 0] * data[:, 4 * j + 4, column], data[0, j, column])
    EXC_i_ampa_input = MPF.I_AMPA(MPF.beta_ampa_input * data[0, 2 * j + 11, column], data[0, j, column])

    return -EXC_i_ampa_local - EXC_i_ampa_input - EXC_i_gaba[0]

# Seed for random number generation
entropy_seed = 12345

# Define input range
INPUTS = np.arange(1, 10, 2) / 100

# Define number of columns and column labels
n_columns = [1, 2]
columns = ["pert", "unpert"]
populations = ["pyr", "inh"]

# Load membrane potentials for wake and NREM states
V_NREM = np.load(DIR + "V_NREM.npy")
V_WAKE = np.load(DIR + "V_WAKE.npy")

# Generate states based on beta_intra and beta_input values
STATES = ["1_1"]
for beta_intra in np.array([2, 4, 6]):
    for beta_input in np.array([2, 4, 6]):
        state = "{}_{}".format(int(beta_intra), int(beta_input))
        STATES.append(state)

# Define noise standard deviations
std_noise_all = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

# Initialize variable for storing standard deviation analysis data
data_std_analysis = None

# Load data for standard deviation analysis
if rank < len(std_noise_all):
    df = pd.read_pickle(DIR + "data_stochastic_firing_True_nColumn_1_bIntra_1.0_bInput_0.0_stdnoise_{}_input_0.0.pkl".format(std_noise_all[rank]))
    data_std_analysis = df["pert"]["pyr"]

# Define beta values for analysis
beta_all = [1.2, 1.6, 2.0, 4.0, 6.0]

# Initialize variable for storing beta analysis data
data_beta_analysis = None

# Load data for beta analysis
if rank < len(beta_all):
    df = pd.read_pickle(DIR + "data_stochastic_firing_True_nColumn_1_bIntra_{}_bInput_0.0_stdnoise_1.2_input_0.0.pkl".format(beta_all[rank]))
    data_beta_analysis = df["pert"]["pyr"]

# Extract beta_intra and beta_input values based on rank
beta_intra = float(STATES[rank].split("_")[0])
beta_input = float(STATES[rank].split("_")[1])

# Initialize lists for storing analysis data
data_spontaneous_analysis = []
data = []

for n_column in n_columns:
    # Load data for spontaneous analysis based on the number of columns
    df = pd.read_pickle(DIR + "data_stochastic_firing_True_nColumn_{}_bIntra_{}_bInput_{}_stdnoise_1.2_input_0.0.pkl".format(n_column, beta_intra, beta_input))
    data_spontaneous_analysis.append(df["pert"]["pyr"])

    # Load data for different input values
    frames = []
    keys = []
    for inpu in INPUTS:
        frames.append(pd.read_pickle(DIR + "data_stochastic_firing_True_nColumn_{}_bIntra_{}_bInput_{}_stdnoise_1.2_input_{}.pkl".format(n_column, beta_intra, beta_input, inpu)))
        keys.append("input:{}".format(inpu))
    df = pd.concat(frames, keys=keys)
    data.append(df)

# Initialize dictionaries for storing analysis results
prestimulus_spontaneous_analysis = {}
deterministic_response = {}
stochastic_amplitude = {}
information_content = {}

# Perform prestimulus analysis if data for standard deviation analysis is available
if data_std_analysis is not None:
    rand_inst = np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(1, int(10 * beta_intra), int(10 * beta_input)))).integers(500)
    prestimulus_std_analysis = FA.PRESTIMULUS_ANALYSIS(data_std_analysis, MPF.dt, rand_inst)
else:
    prestimulus_std_analysis = None

# Perform prestimulus analysis if data for beta analysis is available
if data_beta_analysis is not None:
    rand_inst = np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(2, int(10 * beta_intra), int(10 * beta_input)))).integers(500)
    prestimulus_beta_analysis = FA.PRESTIMULUS_ANALYSIS(data_beta_analysis, MPF.dt, rand_inst)
else:
    prestimulus_beta_analysis = None

for n_column in n_columns:
    # Calculate prestimulus spontaneous analysis
    rand_inst = np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(3, n_column, int(10 * beta_intra), int(10 * beta_input)))).integers(500)
    prestimulus_spontaneous_analysis["{}_column".format(n_column)] = FA.PRESTIMULUS_ANALYSIS(data_spontaneous_analysis[n_column - 1], MPF.dt, rand_inst)

    # Calculate deterministic response for each input value
    deterministic_response["{}_column".format(n_column)] = {}
    binter = beta_input if n_column == 2 else 0
    for ex_input in INPUTS:
        deterministic_response["{}_column".format(n_column)]["input:{}".format(ex_input)] = Deterministic_Response(beta_intra, beta_input, binter, V_WAKE, V_NREM, ex_input)

    # Calculate stochastic amplitude and information content
    stochastic_amplitude["{}_column".format(n_column)] = {}
    information_content["{}_column".format(n_column)] = {}

    if n_column == 1:
        for ex_input in [0.03, 0.07]:
            for times in ["pre", "post"]:
                stochastic_amplitude["{}_column".format(n_column)]["distribution_input:{}_{}".format(ex_input, times)] = data[n_column - 1]["pert"]["input:{}".format(ex_input)]["pyr"][times]

    for i, column in enumerate(columns[:n_column]):
        stochastic_amplitude["{}_column".format(n_column)][column] = {}
        information_content["{}_column".format(n_column)][column] = {}

        for j, pop in enumerate(populations):
            stochastic_amplitude["{}_column".format(n_column)][column][pop] = {}
            information_content["{}_column".format(n_column)][column][pop] = {}

            for times in ["pre", "post"]:
                data_aux_new_arr = np.zeros((2, len(INPUTS)), dtype=float)

                for ii, ex_input in enumerate(INPUTS):
                    data_aux_new = data[n_column - 1][column]["input:{}".format(ex_input)][pop][times]
                    data_aux_new_arr[0, ii] = np.mean(data_aux_new)
                    data_aux_new_arr[1, ii] = np.std(data_aux_new, ddof=1) / np.sqrt(np.size(data_aux_new))

                stochastic_amplitude["{}_column".format(n_column)][column][pop][times] = data_aux_new_arr

            RNG_DETEC = [[np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(0, 0, 2, n_column, i, j, int(10 * beta_intra), int(10 * beta_input), k, l),)) for k in range(len(INPUTS))] for l in range(2)]
            RNG_DIFF = [np.random.default_rng(np.random.SeedSequence(entropy=entropy_seed, spawn_key=(0, 0, 3, n_column, i, j, int(10 * beta_intra), int(10 * beta_input), k),)) for k in range(2)]

            info_brain_state = FA.INFORMATION_BRAIN_STATE(RNG_DETEC, RNG_DIFF, data[n_column - 1], INPUTS, column, pop)

            information_content["{}_column".format(n_column)][column][pop]["detec_logistic"] = info_brain_state.info_detec_logistic
            information_content["{}_column".format(n_column)][column][pop]["diff_logistic"] = info_brain_state.info_diff_logistic
            information_content["{}_column".format(n_column)][column][pop]["detec_k_means"] = info_brain_state.info_detec_k_means
            information_content["{}_column".format(n_column)][column][pop]["diff_k_means"] = info_brain_state.info_diff_k_means
            information_content["{}_column".format(n_column)][column][pop]["tvalue"] = info_brain_state.tvalue
            information_content["{}_column".format(n_column)][column][pop]["fvalue"] = info_brain_state.fvalue
            information_content["{}_column".format(n_column)][column][pop]["mi"] = info_brain_state.mi
            information_content["{}_column".format(n_column)][column][pop]["mi_detect"] = info_brain_state.mi_detect

# Gather analysis results from all processes to root process
prestimulus_std_analysis = comm.gather(prestimulus_std_analysis, root=0)
prestimulus_beta_analysis = comm.gather(prestimulus_beta_analysis, root=0)
prestimulus_spontaneous_analysis = comm.gather(prestimulus_spontaneous_analysis, root=0)
deterministic_response = comm.gather(deterministic_response, root=0)
stochastic_amplitude = comm.gather(stochastic_amplitude, root=0)
information_content = comm.gather(information_content, root=0)

# Root process saves gathered data
if rank == 0:
    data_to_save = {
        "prestimulus_std_analysis": {},
        "prestimulus_beta_analysis": {},
        "prestimulus_spontaneous_analysis": {},
        "deterministic_response": {},
        "stochastic_amplitude": {},
        "information_content": {}
    }

    for i, std_noise in enumerate(std_noise_all):
        data_to_save["prestimulus_std_analysis"]["std:{}".format(std_noise)] = prestimulus_std_analysis[i]

    for i, beta in enumerate(beta_all):
        data_to_save["prestimulus_beta_analysis"]["beta:{}".format(beta)] = prestimulus_beta_analysis[i]

    for i, state in enumerate(STATES):
        data_to_save["prestimulus_spontaneous_analysis"][state] = prestimulus_spontaneous_analysis[i]
        data_to_save["deterministic_response"][state] = deterministic_response[i]
        data_to_save["stochastic_amplitude"][state] = stochastic_amplitude[i]
        data_to_save["information_content"][state] = information_content[i]

    # Additional analysis for specific states and inputs
    ex_input = 0.05
    for state in ["2_2", "2_6", "6_2"]:
        bintra = int(state.split("_")[0])
        binput = int(state.split("_")[1])
        binter = 0
        EI_record = Obtain_EI_Trace(bintra, binput, binter, V_WAKE, ex_input)
        data_to_save["deterministic_time_trace_E_I_one_column_input_0.05"][state] = EI_record

    with open(DIR + 'data_curation.pickle', 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Ensure variables are None in non-root processes
    assert prestimulus_std_analysis is None
    assert prestimulus_beta_analysis is None
    assert prestimulus_spontaneous_analysis is None
    assert deterministic_response is None
    assert stochastic_amplitude is None
    assert information_content is None
