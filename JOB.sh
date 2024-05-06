#!/bin/bash
##############################################################
#                            BASH JOB                        #
#                                                            #
#                                                            #
##############################################################
# 
# 
# 
# 
# home address
DIRECTORY="/home/frazi/"
export DIRECTORY
#
#
#
# store job id
JOBS=""
# 
# 
# 
# 
echo >> ./textfileme.txt
echo >> ./textfileme.txt
echo "##############################################################################################################################################" >> ./textfileme.txt
echo "############################################################### Job is started ###############################################################" >> ./textfileme.txt
echo "##############################################################################################################################################" >> ./textfileme.txt
#
#
#
#
##############################################################
#                          1st Block                         #
#                          NREM Sleep                        #
##############################################################
Beta_Intra=1 Beta_Input=0 Beta_Inter=0 STOCHASTIC=1 SPONTANEOUS=1 STD_NOISE=1.2
jobname="${Beta_Intra}_${Beta_Input}_${Beta_Inter}_${STD_NOISE}_${STOCHASTIC}_${SPONTANEOUS}"
jid1=$(sbatch -J ${jobname} --mem=8G --array=0 --time=0-12:00:00 ./slurm_scripts/SIMULATION.sbatch "${DIRECTORY}" ${Beta_Intra} ${Beta_Input} ${Beta_Inter} ${STD_NOISE} ${STOCHASTIC} ${SPONTANEOUS})
JOBS="${JOBS}${jid1##* },"
#
#
#
#
##############################################################
#                          2st Block                         #
# Robustness to the standard deviation of the Gaussian noise #
##############################################################
Beta_Intra=1 Beta_Input=0 Beta_Inter=0 STOCHASTIC=1 SPONTANEOUS=1
STD_NOISES="0.9 1.0 1.1 1.3 1.4"
for i in ${STD_NOISES};do
    STD_NOISE=${i}
    jobname="${Beta_Intra}_${Beta_Input}_${Beta_Inter}_${STD_NOISE}_${STOCHASTIC}_${SPONTANEOUS}"
    jid2=$(sbatch -J ${jobname} --mem=8G --array=0 --time=0-12:00:00 ./slurm_scripts/SIMULATION.sbatch "${DIRECTORY}" ${Beta_Intra} ${Beta_Input} ${Beta_Inter} ${STD_NOISE} ${STOCHASTIC} ${SPONTANEOUS})
    JOBS="${JOBS}${jid2##* },"
done
#
#
#
#
##############################################################
#                          3rd Block                         #
#       Synaptic upscaling shifts the dynamics to wake       #
##############################################################
Beta_Input=0 Beta_Inter=0 STOCHASTIC=1 SPONTANEOUS=1
Beta_IntraS="1.2 1.6 2.0 4.0 6.0"
for i in ${Beta_IntraS};do
    Beta_Intra=${i}
    jobname="${Beta_Intra}_${Beta_Input}_${Beta_Inter}_${STD_NOISE}_${STOCHASTIC}_${SPONTANEOUS}"
    jid2=$(sbatch --dependency=aftercorr:${jid1##* } -J ${jobname} --mem=8G --array=0 --time=0-12:00:00 ./slurm_scripts/SIMULATION.sbatch "${DIRECTORY}" ${Beta_Intra} ${Beta_Input} ${Beta_Inter} ${STD_NOISE} ${STOCHASTIC} ${SPONTANEOUS})
    JOBS="${JOBS}${jid2##* },"
done
#
#
#
#
#
#
#
##############################################################
#                          4th Block                         #
#                  Stochastic Evoked responses               #
##############################################################
STATES="1_1"
Beta_IntraS="2 4 6"
Beta_InputS="2 4 6"
for Beta_Intra in $Beta_IntraS;do 
    for Beta_Input in $Beta_InputS; do 
        STATES="${STATES} ${Beta_Intra}_${Beta_Input}"
    done;
done

Beta_Inter=0 STD_NOISE=1.2 STOCHASTIC=1 SPONTANEOUS=0
for state in ${STATES}; do
    Beta_Intra=${state%_*} Beta_Input=${state#*_}
    jobname="${Beta_Intra}_${Beta_Input}_${Beta_Inter}_${STD_NOISE}_${STOCHASTIC}_${SPONTANEOUS}"
    jid2=$(sbatch --dependency=aftercorr:${jid1##* } -J ${jobname} --array=1,3,5,7,9,11 --time=0-07:00:00 ./slurm_scripts/SIMULATION.sbatch "${DIRECTORY}" ${Beta_Intra} ${Beta_Input} ${Beta_Inter} ${STD_NOISE} ${STOCHASTIC} ${SPONTANEOUS})
    JOBS="${JOBS}${jid2##* },"
done
#
#
#
#
#
#
#
##############################################################
#                          5th Block                         #
#         NREM Sleep and Wake in two-cortical column         #
##############################################################
STD_NOISE=1.2 STOCHASTIC=1 SPONTANEOUS=1
for state in ${STATES}; do
    Beta_Intra=${state%_*} Beta_Input=${state#*_} Beta_Inter=${state#*_}
    jobname="${Beta_Intra}_${Beta_Input}_${Beta_Inter}_${STD_NOISE}_${STOCHASTIC}_${SPONTANEOUS}"
    jid2=$(sbatch --dependency=aftercorr:${jid1##* } -J ${jobname} --mem=8G --array=0 --time=0-12:00:00 ./slurm_scripts/SIMULATION.sbatch "${DIRECTORY}" ${Beta_Intra} ${Beta_Input} ${Beta_Inter} ${STD_NOISE} ${STOCHASTIC} ${SPONTANEOUS})
    JOBS="${JOBS}${jid2##* },"
done

#
#
#
#
#
#
##############################################################
#                          6th Block                         #
#    Stochastic Evoked responses in two-cortical column      #
##############################################################
STD_NOISE=1.2 STOCHASTIC=1 SPONTANEOUS=0
for state in ${STATES}; do
    Beta_Intra=${state%_*} Beta_Input=${state#*_} Beta_Inter=${state#*_}
    jobname="${Beta_Intra}_${Beta_Input}_${Beta_Inter}_${STD_NOISE}_${STOCHASTIC}_${SPONTANEOUS}"
    jid2=$(sbatch --dependency=aftercorr:${jid1##* } -J ${jobname} --array=1,3,5,7,9,11 --time=0-07:00:00 ./slurm_scripts/SIMULATION.sbatch "${DIRECTORY}" ${Beta_Intra} ${Beta_Input} ${Beta_Inter} ${STD_NOISE} ${STOCHASTIC} ${SPONTANEOUS})
    JOBS="${JOBS}${jid2##* },"
done
#
#
#
#
#
#
#
#
##############################################################
#                          7th Block                         #
#                          Analysis                          #
##############################################################
jid3=$(sbatch --dependency=aftercorr:${JOBS%%,} -J analysis --mem=25G -N 1 -n 10 ./slurm_scripts/ANALYSIS.sbatch "${DIRECTORY}")
#
#
#
#
#
#
#
#
##############################################################
#                          8th Block                         #
#                            Plot                            #
##############################################################
sbatch --dependency=aftercorr:${jid3##* } -J plot --mem=25G ./slurm_scripts/PLOT.sbatch "${DIRECTORY}"
#
#
#
#
#
#
echo >> ./textfileme.txt
echo >> ./textfileme.txt
echo "##############################################################################################################################################" >> ./textfileme.txt
echo "############################################################### Job is finished ##############################################################" >> ./textfileme.txt
echo "##############################################################################################################################################" >> ./textfileme.txt
