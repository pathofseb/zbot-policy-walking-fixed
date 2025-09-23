#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name="zbot-walking-training"
#SBATCH --time=02:00:00
#SBATCH --partition=GPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="(a100|h100|h200)&(gpu40g|gpu80g)"
#SBATCH --output=slurm_outputs/zbot_training_%j.out
#SBATCH --error=slurm_outputs/zbot_training_%j.err
#SBATCH --mail-user=sebasrb@ntnu.no
#SBATCH --mail-type=ALL

# Create output directory if it doesn't exist
mkdir -p slurm_outputs

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

echo "========================================"
echo "ZBot Walking Policy Training"
echo "========================================"
echo "Running from directory: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"

# Load necessary modules (adjust based on your cluster)
# module load Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import jax; print(f'JAX backend: {jax.default_backend()}'); print(f'JAX devices: {jax.devices()}')"

# Install requirements if needed (uncomment if running for first time)
# echo "Installing requirements..."
# pip install -r requirements.txt

# Start training
echo "Starting ZBot training..."
echo "Configuration: 2048 environments, batch size 256"
echo "Expected training time: ~30-60 minutes for walking behavior"

# Run the training
python -m train

echo "========================================"
echo "Training completed at: $(date)"
echo "Check TensorBoard logs in: zbot_walking_task/"
echo "Trained model saved in: zbot_walking_task/run_XXX/checkpoints/"
echo "========================================"