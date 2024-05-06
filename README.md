# heterogeneous_synaptic_homeostasis

### Introduction
This GitHub repository contains scripts for implementing heterogeneous synaptic homeostasis, a novel synaptic mechanism designed to elucidate shifts in the propagation pattern of information across the cortex during the sleep-wake cycle. Detailed information on this mechanism can be found in [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1).

## Repository Structure
```
heterogeneous_synaptic_homeostasis
├── data
│   └── data.zip
│       ├── data_curation.pickle
│       ├── V_NREM.npy
│       └── V_WAKE.npy
├── figures
│   └── figures.zip
│       └── SVG files of the figures
├── python_scripts
│   ├── model_parameters_functions.py
│   ├── NUMERICAL_SIMULATION.py
│   ├── functions_analysis.py
│   ├── ANALYSIS_MPI.py
│   └── PLOT_FIGURES.py
├── slurm_scripts
│   ├── SIMULATION.sbatch
│   ├── ANALYSIS.sbatch
│   └── PLOT.sbatch
├── JOB.sh
├── LICENSE
└── README.md
```

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of JOB.sh
To compile this code on high-performance computing systems, ensure that you provide the correct directory address in the **_JOB.sh**_ script. The **_JOB.sh**_ script is an executable script that queues the simulations on the Slurm Workload Manager. Please note that you can implement simulations on your local machine using only Python scripts as well. In the latter case, you have to provide the right variables described in the **Description of python scripts** section.

You can run the **_JOB.sh**_ script by executing the following commands in the terminal:
```
$ chmod +x JOB.sh
$ ./JOB.sh
```

Briefly, the script is organized into blocks, each corresponding to specific simulations. The 1st, 2nd, 3rd, and 4th blocks correspond to simulations of 500 stochastic trials in the one-cortical column model. The 5th and 6th blocks correspond to simulations of 500 stochastic trials in the two-cortical column model. Finally, the 7th and 8th blocks correspond to the analysis and plotting of the figures, respectively.

1. The first block simulates independent spontaneous trials for NREM sleep in the one-cortical column model. There are two ways to simulate spontaneous trials: Beta_Input=0 and/or --array=0 (where --array is the argument for the sbatch script determining the intensity of the external stimulus). The average firing rate of pyramidal and inhibitory populations during the Up state of these 500 trials is used to determine the steady-state values of the model during wakefulness states. These values regulate inhibition to counterbalance overexcitation due to synaptic upscaling during wakefulness. The same steady-state values are used for wakefulness in the two cortical column model. Additionally, the steady-state of the model during NREM sleep in one cortical column model is numerically calculated and used for the steady-state value of the two-cortical column model during NREM sleep. These steady-state values are utilized to determine the amount of regulation in inhibition due to inter-excitatory connections between the two columns. These values for the present study are provided in the data directory as _V_NREM.npy_ and _V_WAKE.npy_. Note that the Slurm jobs from the 3rd Block and after depend on the success of the first block in determining the steady-state values.

2. The second block simulates independent spontaneous trials for NREM sleep when the standard deviation of the noise in the model is varied by up to 10%.

3. The third block simulates independent spontaneous trials for wakefulness when the intra-synaptic upscaling factor $`\beta_{\text{intra}}`$ increases from 1.2 to 6.

4. The fourth block simulates independent evoked trials for NREM sleep and wakefulness when the intra- and inter-synaptic upscaling factors ($`\beta_{\text{intra}}`$ and $`\beta_{\text{inter}}`$, respectively) increase from 2 to 6. The intensity of stimuli is specified as the --array arguments in the sbatch script.

5. The fifth block simulates independent spontaneous trials for NREM sleep and wakefulness when the intra- and inter-synaptic upscaling factors ($`\beta_{\text{intra}}`$ and $`\beta_{\text{inter}}`$, respectively) increase from 2 to 6.

6. The sixth block simulates independent evoked trials for NREM sleep and wakefulness when the intra- and inter-synaptic upscaling factors ($`\beta_{\text{intra}}`$ and $`\beta_{\text{inter}}`$, respectively) increase from 2 to 6. The intensity of stimuli is specified as the --array arguments in the sbatch script.

7. The seventh block corresponds to the analysis carried out for the purpose of the article [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1). It utilizes mpi4py for parallelizing Python scripts across 10 processing units for the analysis.

8. Finally, the eighth block corresponds to plotting figures for the same purpose of the article [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1).

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of sbatch scripts

The sbatch scripts distribute simulations on the Slurm Workload Manager. They are located within the _slurm_scripts_ directory and include:
- _SIMULATION.sbatch_ 
- _ANALYSIS.sbatch_
- _PLOT.sbatch_
Please ensure you provide the correct module to load in each sbatch script within the slurm_scripts directory, and also remember to update the email address in the --mail-user argument of the sbatch scripts accordingly.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_SIMULATION.sbatch_:** This script is aimed at implementing the NUMERICAL_SIMULATION.py script in the python_scripts directory for numerical simulations of the model. There are seven arguments necessary to provide:
- DIRECTORY: The directory of the repository on the machine.
- Beta_Intra: The values for synaptic upscaling of intra-synaptic connections.
- Beta_Input: The values for synaptic upscaling of synaptic connections where stimulus is applied.
- Beta_Inter: The values for synaptic upscaling of inter-synaptic connections.
- STD_NOISE: The standard deviation of the Gassuian noise.
- STOCHASTIC: A binary value to determine if simulations are stochastic (STOCHASTIC=1) or deterministic (STOCHASTIC=0).
- SPONTANEOUS: A binary value to determine if simulations are spontaneous (SPONTANEOUS=1) or deterministic (SPONTANEOUS=0).

Additionally, it assigns a job array value that corresponds to the intensity of the stimuli applied to the model in the _NUMERICAL_SIMULATION.py_ script.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_ANALYSIS.sbatch_:** This script is aimed at implementing the _ANALYSIS_MPI.py_ script for the analysis of the simulations. The only necessary argument to provide is the DIRECTORY.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_PLOT.sbatch_:** This script is aimed at implementing the _PLOT_FIGURES.py_ script for plotting figures for the purpose of the following article [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1). The only necessary argument to provide is the DIRECTORY.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------


### Description of python scripts
The Python scripts are located within the _python_scripts_ directory and include:
- _model_parameters_functions.py_ 
- _NUMERICAL_SIMULATION.py_
- _functions_analysis.py_
- _ANALYSIS_MPI.py_ 
- _PLOT_FIGURES.py_

**Important Note:** If you wish to implement simulations on your local machine using only Python scripts, you have to provide the right variables described here either as environment variables or by changing them manually within the Python scripts.
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_model_parameters_functions.py_:** This script is aimed at assigning the model parameters and defining the functions regarding numerical integrations.

**Python requirements:**
- numpy

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_NUMERICAL_SIMULATION.py_:** This script is aimed at implementing the numerical integrations for a given set of variables. There are eight arguments necessary to provide:
- DIR: The directory of the repository on the machine, provided by the **DIRECTORY** environment variable.
- MPF.beta_ampa_intra: The values for synaptic upscaling of intra-synaptic connections, provided by the **Beta_Intra** environment variable.
- MPF.beta_ampa_input: The values for synaptic upscaling of synaptic connections where stimulus is applied, provided by the **Beta_Input** environment variable.
- MPF.beta_ampa_inter: The values for synaptic upscaling of inter-synaptic connections, provided by the **Beta_Inter** environment variable.
- MPF.phi_n_sd: The standard deviation of the Gaussian noise, provided by the **STD_NOISE** environment variable.
- stochastic: A boolean value to determine if simulations are stochastic (stochastic=True) or deterministic (stochastic=False), provided by the **STOCHASTIC** environment variable.
- spontaneous: A boolean value to determine if simulations are stochastic (spontaneous=True) or deterministic (spontaneous=False), provided by the **SPONTANEOUS** environment variable.
- ex_input: The intensity of the stimuli provided by the **SLURM_ARRAY_TASK_ID** environment variable divided by 100.


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

**_functions_analysis.py_:** This script aims to define functions for analyzing both the spontaneous characteristics of firing rate signals and the information content carried by evoked firing signals in response to stimuli.

**Python requirements:**
- numpy
- scipy
- sklearn
- pandas
- pickle
- sys

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

**_ANALYSIS_MPI.py_:** This script is aimed at carrying out the analysis by parallelizing jobs among processing units. The only necessary argument to provide is the DIRECTORY.

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

**_PLOT_FIGURES.py_:** This script is aimed at plotting figures for the purpose of the following article [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1). The only necessary argument to provide is the DIRECTORY.

**Python requirements:**
- numpy
- scipy
- pickle
- pandas
- os
- matplotlib
- mpl_toolkits
- seaborn

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of data
This directory includes a zip file containing processed data (_data_curation.pickle_) and the steady-state values for NREM sleep (_V_NREM_) and wakefulness (_V_WAKE_). These steady-state values determine the regulation of inhibition to counterbalance the overexcitation due to synaptic upscaling and inter-synaptic connections in the model.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

### Description of figures

This directory contains a zip file containing figures referenced in the following article: [doi:10.1101/2023.12.04.569905](https://www.biorxiv.org/content/10.1101/2023.12.04.569905v1).

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contributing

Thank you for considering contributing to our project! We welcome contributions from the community to help improve our project and make it even better. To ensure a smooth contribution process, please follow these guidelines:

1. **Fork the Repository**: Fork our repository to your GitHub account and clone it to your local machine.

2. **Branching Strategy**: Create a new branch for your contribution. Use a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow our coding standards and style guidelines. Make sure your code adheres to the existing conventions to maintain consistency across the project.

4. **Pull Request Process**:
    Before starting work, check the issue tracker to see if your contribution aligns with any existing issues or feature requests.
    Create a new branch for your contribution and make your changes.
    Commit your changes with clear and descriptive messages explaining the purpose of each commit.
    Once you are ready to submit your changes, push your branch to your forked repository.
    Submit a pull request to the main repository's develop branch. Provide a detailed description of your changes and reference any relevant issues or pull requests.

5. **Code Review**: Expect feedback and review from our maintainers or contributors. Address any comments or suggestions provided during the review process.

6. **Testing**: Ensure that your contribution is properly tested. Write unit tests or integration tests as necessary to validate your changes. Make sure all tests pass before submitting your pull request.

7. **Documentation**: Update the project's documentation to reflect your changes. Include any necessary documentation updates, such as code comments, README modifications, or user guides.

8. **License Agreement**: By contributing to our project, you agree to license your contributions under the terms of the project's license (GNU General Public License v3.0).

9. **Be Respectful**: Respect the opinions and efforts of other contributors. Maintain a positive and collaborative attitude throughout the contribution process.

We appreciate your contributions and look forward to working with you to improve our project! If you have any questions or need further assistance, please don't hesitate to reach out to us.

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Credits

- **Author:** [Farhad Razi](https://github.com/fraziphy)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE)

------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------

## Contact

- **Contact information:** [email](farhad.razi.1988@gmail.com)
