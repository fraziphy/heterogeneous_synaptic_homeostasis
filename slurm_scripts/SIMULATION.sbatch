#!/bin/bash
##############################################################
#                                                            #
#                                                            #
#                                                            #
#                                                            #
#                          SLURM Jobs                        #
#                                                            #
#                                                            #
#                                                            #
#                                                            #
##############################################################
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=farhad.razi01@estudiant.upf.edu
#SBATCH --output=output_run_%x_%a.out
#SBATCH --error=error_run_%x_%a.err
# 
# 
# 
#
# 
module load CuPy/8.2.0-foss-2020b-Python-3.8.6
export DIRECTORY=$1 Beta_Intra=$2 Beta_Input=$3 Beta_Inter=$4 STD_NOISE=$5 STOCHASTIC=$6 SPONTANEOUS=$7
# 
# 
# 
python3 ../python_scripts/NUMERICAL_SIMULATION.py
