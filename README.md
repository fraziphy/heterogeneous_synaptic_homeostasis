# heterogeneous_synaptic_homeostasis

### This GitHub repository includes scripts for implementing heterogeneous synaptic homeostasis, a novel synaptic mechanism aimed at elucidating shifts in the propagation pattern of information across the cortex during the sleep-wake cycle, detailed in [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1).

## The structure of the repository is as follows:
```
iQuanta
├── data
│   ├── processed
│   │   └── my_file.pkl
│   └── raw
│       └── my_file.pkl
├── iQuanta
│   ├── __init__.py
│   ├── config.py
│   └── funcs.py
├── notebook
│   └── iQuanta.ipynb
├── scripts
│   ├── __init__.py
│   ├── config.py
│   ├── generate_raw_data.py
│   ├── plot_figures.py
│   └── process_data.py
├── tests
│   ├── test_config.py
│   └── test_funcs.py
├── LICENSE
├── README.md
└── setup.py
```

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of JOB.sh

To compile this code on high performance computing systems, make sure you provide the rigth directory address in the JOB.sh script. JOB.sh script is an executable script that que the simulations on Slurm Workload Manager.

Briefly, 1st, 2nd, 3rd, and the 4th Blocks corresponds to simulations of 500 stocastic trials in the one-cortical column model. The 5th and 6th Blocks corresponds to simulations of 500 stocastic trials in the tw-cortical column model. Finally, the 7th Block corresponds to the analysis.

The 1st Block simulate independent spontaneous trials for NREM sleep in one-cortical column model. Note that there are two ways to simulate spontaneous trials: Beta_Input=0 and/or array=0 (the array is the argument for the sbatch script and it determines the intensity of the external stimulus). The average firing rate of pyramidal and inhibitory populations during Up state of these 500 trials are used to determine the steady-state values of the model during wakefulness states. Then, these values are used to detremine the regulation of inhibiton to counterbalance the overexcitation due to synaptic upscaling during wakefulness. This procedure garanntee that synaptuc upscaling during wakeuflenss does not lead to overexcitation or over inhibition. The same values of steady states are used for the wakefulness in the two cortical column model. Additionaly, the steady-state of the model during NREM sleep in one cortical column model is numericaly calculated and is going to be used for the steady-state value of two-cortical column model during NREM sleep. These steady-state values are utelized to determine the amount of regulation in inhibition due to inter-excitatory connections between the two coulmns. These values for the present study are provided in _data_ directory as _NREM.npy_ and _WAKE.npy_. Please note that the slurm jobs from the 3rd Blocks and after depends on the success of the first block determining the steady-state values.

The 2nd Block simulate independent spontaneous trials for NREM sleep when the standard deviation of the noise in the model is varied by up to 10%. The 3rd Block simulate independent spontaneous trials for wakefulness when the intra-synaptic upscaling factor $`\beta_{\text{intra}}`$ increases from 1.2 to 6. The 4th Block simulate independent evoked trials for NREM sleep and wakefulness when the intra- and inter-synaptic upscaling factors ($`\beta_{\text{intra}}`$ and $`\beta_{\text{inter}}`$, respectively) increases from 2 to 6. The intensity of stimuli are specified as the --array arguments in the sbatch script.

The 5th Block simulate independent spontaneous trials for NREM sleep and wakefulness when the intra- and inter-synaptic upscaling factors ($`\beta_{\text{intra}}`$ and $`\beta_{\text{inter}}`$, respectively) increases from 2 to 6. The 6th Block simulate independent evoked trials for NREM sleep and wakefulness when the intra- and inter-synaptic upscaling factors ($`\beta_{\text{intra}}`$ and $`\beta_{\text{inter}}`$, respectively) increases from 2 to 6. The intensity of stimuli are specified as the --array arguments in the sbach script.

The 7th Block corresponds to the analysis that are carried out for the purpose of the following article [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1). It utelizes mpi4py for parallelizing Python scripts across 10 processing units for the analysis. Finally, the 8th Block corresponds to plotting figures for the purpose of the following article [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1).

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

```math
SE = \frac{\sigma}{\sqrt{n}}
```

### Description of sbatch scripts

The sbatch scripts distribute simulations on Slurm Workload Manager. The sbatch scripts are located within the _slurm_scripts_ directory, including:
- _SIMULATION.sbatch_ 
- _ANALYSIS.sbatch_
- _PLOT.sbatch_
Please, make sure you provide the correct module to load in each sbatch script in within slurm_scripts. please make sure to change the email address in the --mail-user argument of the sbatch scripts accordingly.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_SIMULATION.sbatch_** is aimed to impliment the _NUMERICAL_SIMULATION.py_ script in the _python_scripts_ directory for numerical simulations of the model. There are seven argument necessary to privide as:
- DIRECTORY: The directory of the repository on the machine,
- Beta_Intra: The values for synaptic upscaling of intra-synaptic connections
- Beta_Input: The values for synaptic upscaling of synaptic connections where stimulus is applied
- Beta_Inter: The values for synaptic upscaling of inter-synaptic connections
- STD_NOISE: The standard deviation of the Gassuian noise
- STOCHASTIC: A binary value to determine if simulations are stockastic (STOCHASTIC=1) or deterministic (STOCHASTIC=0)
- SPONTANEOUS: A binary value to determine if simulations are stockastic (SPONTANEOUS=1) or deterministic (SPONTANEOUS=0)

Additinally, it assign a job array value that corresponds to the intensity of the stimuli that are appiled to the model in the _NUMERICAL_SIMULATION.py_ script.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_ANALYSIS.sbatch_** is aimed to impliment the _ANALYSIS_MPI.py_ script for analysis of the simulations. The only argument neccessary to provide is the DIRECTORY.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_PLOT.sbatch_** is aimed to impliment the _PLOT_FIGURES.py_ script for plotting figures of the simulations. The only argument neccessary to provide is the DIRECTORY.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------


### Description of python scripts
The Python scripts are located within the _python_scripts_ directory, including:
- _model_parameters_functions.py_ 
- _NUMERICAL_SIMULATION.py_
- _functions_analysis.py_
- _ANALYSIS_MPI.py_ 
- _PLOT_FIGURES.py_

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_model_parameters_functions.py_** is aimed to assign the model parameters and definne the functions regarding numerical integrations.
Python requirements:
- numpy

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_NUMERICAL_SIMULATION.py_** is aimed to impliment the numerical integrations for a given set of variables. There are eight argument necessary to privide as:
- DIR: The directory of the repository on the machine,
- MPF.beta_ampa_intra: The values for synaptic upscaling of intra-synaptic connections
- MPF.beta_ampa_input: The values for synaptic upscaling of synaptic connections where stimulus is applied
- MPF.beta_ampa_inter: The values for synaptic upscaling of inter-synaptic connections
- MPF.phi_n_sd: The standard deviation of the Gassuian noise
- stochastic: A boolian value to determine if simulations are stockastic (stochastic=True) or deterministic (stochastic=False)
- spontaneous: A boolian value to determine if simulations are stockastic (spontaneous=True) or deterministic (spontaneous=False)
- ex_input: The intensity of the stimuli provided by the SLURM_ARRAY_TASK_ID enviroment variable divided by 100

**Modeling one-cortical column:**
- NREM: MPF.beta_ampa_intra = 1, MPF.beta_ampa_inter = 0
- Wake: MPF.beta_ampa_intra => 2, MPF.beta_ampa_inter = 0
- _Spontaneous activity_:
    - NREM & Wake: MPF.beta_ampa_input = 0 (or MPF.beta_ampa_input != 0 and ex_input=0)
- _Evoked activity_:
    - NREM: MPF.beta_ampa_input = 1
    - Wake: MPF.beta_ampa_input => 2

**Modeling two-cortical column:**
- NREM: MPF.beta_ampa_intra = 1, MPF.beta_ampa_inter = 1
- Wake: MPF.beta_ampa_intra => 2, MPF.beta_ampa_inter => 2 (the values for intra and inter are not chosen independently, i.e., they can get different values)
- _Spontaneous activity_:
    - NREM & Wake: MPF.beta_ampa_input = 0 (or MPF.beta_ampa_input != 0 and ex_input=0)
- _Evoked activity_:
    - NREM: MPF.beta_ampa_input = 1
    - Wake: MPF.beta_ampa_input => 2 (In our study we set MPF.beta_ampa_input=MPF.beta_ampa_inter to avoid sturation of evoked responses)

**Python requirements:**
- _model_parameters_functions.py_
- numpy
- pandas
- pickle
- os

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_functions_analysis.py_** is aimed to definne the functions regarding analysis. It includes analysis of spontaneous charecteristics of firing rate signals and information content that evoked firing signals to stimuli carry about.

**Python requirements:**
- numpy
- scipy
- sklearn
- pandas
- pickle
- sys

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**- _ANALYSIS_MPI.py_** is aimed to carry out the analysis by parralyzing jobs among processing units. The only argument neccessary to provide is the DIRECTORY.
**Python requirements:**
- functions_analysis
- model_parameters_functions
- numpy
- pickle
- pandas
- os
- mpi4py

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**- _PLOT_FIGURES.py_** is aimed to carry out the analysis by parralyzing jobs among processing units. The only argument neccessary to provide is the DIRECTORY.
**Python requirements:**
- functions_analysis
- model_parameters_functions
- numpy
- pickle
- pandas
- os
- mpi4py
