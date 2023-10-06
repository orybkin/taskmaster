#! /bin/bash

echo $SLURM_JOB_ID
echo slurm job id
singularity run -B /var/lib/dcv-gl --nv --writable-tmpfs  --bind /global/scratch/users/oleh/tmp:/tmp /global/scratch/users/oleh/taskmaster.sif -- bash /global/scratch/users/oleh/taskmaster/isaacgymenvs/run_container.sh  slurm_job_id=$SLURM_JOB_ID "$@"