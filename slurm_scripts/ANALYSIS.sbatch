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
#SBATCH --time=0-7:00:00
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
module load scikit-learn/0.23.2-foss-2020b
module load geopandas/0.10.2-foss-2020b-Python-3.8.6
module load SciPy-bundle/2021.10-foss-2020b
module load pybind11/2.7.1-GCCcore-10.2.0
export DIRECTORY=$1
# 
# 
# 
mpirun -np $SLURM_NTASKS python3 ../python_scripts/ANALYSIS_MPI.py
