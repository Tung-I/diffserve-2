#!/bin/bash
#SBATCH --job-name=feat 
#SBATCH -N1                          # Ensure that all cores are on one machine
#SBATCH --partition=gpupod-l40s             # Partition to submit to (serial_requeue)
#SBATCH --mem=32768               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/work/pi_huiguan_umass_edu/qizhengyang/query_aware_pipe/extend_diffserve/router/oslog/run_logs_%j.out            # File to which STDOUT will be written
#SBATCH --error=/work/pi_huiguan_umass_edu/qizhengyang/query_aware_pipe/extend_diffserve/router/oslog/run_logs_%j.err            # File to which STDERR will be written
#SBATCH -q gpu-quota-20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=qizhengyang@umass.edu

echo `pwd`
# echo "SLURM task ID: "$SLURM_ARRAY_TASK_ID
#module load cuda/11.3.1
set -x -e
##### Experiment settings #####
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh" ]; then
        . "/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/etc/profile.d/conda.sh"
    else
        export PATH="/modules/apps/miniconda/4.8.3/envs/jupyterhub-stable/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<
conda init bash
conda activate /work/qizhengyang_umass_edu/.conda/envs/morph
sleep 1

cd /work/pi_huiguan_umass_edu/qizhengyang/query_aware_pipe/extend_diffserve/router
python ./ruleBasedTest.py ;

sleep 1
exit
