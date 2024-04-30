#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -p a100
#SBATCH --gpus-per-node=1


mini_env=PyTorch_1.13_n


source $HOME/miniconda3/bin/activate
conda activate ${mini_env}
module load cuda/11.6
module load gcc/10.3.0
module load cmake/3.22.2
export PYTHONNOUSERSITE=true

export CUDA_HOME=/home/gulhane.2/cuda/setup.sh

export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH


batch_size=1
parts=1
image_size=2048
precision="fp_16"

OUTFILE="logs/benchmark_resnet_single_gpu_inference.log"

python ../benchmarks/single_gpu_inference/benchmark_resnet_inference.py \
        --batch-size ${batch_size} \
        --image-size ${image_size} \
        --precision ${precision} \
        --enable-evaluation &>> $OUTFILE 2>&1
