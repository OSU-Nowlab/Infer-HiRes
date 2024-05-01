# Run Layer Parallelism Inference


#### Generic command:
```bash

mpirun_rsh --export-all -np $np\
        --hostfile ${hostfile}  \
        python layer_parallelism/benchmark_resnet_lp.py \
        --batch-size ${batch_size} \
        --split-size ${split_size} \
        --parts ${parts} \
        --image-size ${image_size} \
        --backend ${backend} \
        --precision ${precision} \
        --checkpoint ${checkpoint_path} \
        --datapath ${datapath} \
        --enable-evaluation

```
#### Examples

- With 4 GPUs [split size: 4]

Example to run ResNet LP Inference with 4 model split size(i.e. # of partitions for LP) for 1024 * 1024 image size.

```bash
mpirun_rsh --export-all -np $np\
        --hostfile ${hostfile}  \
        MV2_USE_CUDA=1 \
        MV2_HYBRID_BINDING_POLICY=spread \
        MV2_CPU_BINDING_POLICY=hybrid \
        MV2_USE_GDRCOPY=0 \
        PYTHONNOUSERSITE=true \
        LD_PRELOAD=$MV2_HOME/lib/libmpi.so \
        python layer_parallelism/benchmark_resnet_lp.py \
        --batch-size 1 \
        --split-size 4 \
        --parts 1 \
        --image-size 1024 \
        --backend "nccl" \
        --precision "fp_16" \
        --enable-evaluation
```



Below are the available configuration options :

<pre>
optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Prints performance numbers or logs (default: False)
  --batch-size BATCH_SIZE
                        input batch size (default: 32)
  --parts PARTS         Number of parts for MP (default: 1)
  --split-size SPLIT_SIZE
                        Number of process for MP (default: 2)
  --num-spatial-parts NUM_SPATIAL_PARTS
                        Number of partitions in spatial parallelism (default: 4)
  --spatial-size SPATIAL_SIZE
                        Number splits for spatial parallelism (default: 1)
  --times TIMES         Number of times to repeat MASTER 1: 2 repications, 2: 4 replications (default: 1)
  --image-size IMAGE_SIZE
                        Image size for synthetic benchmark (default: 32)
  --num-layers NUM_LAYERS
                        Number of layers in amoebanet (default: 18)
  --num-classes NUM_CLASSES
                        Number of classes (default: 10)
  --balance BALANCE     length of list equals to number of partitions and sum should be equal to num layers (default: None)
  --halo-D2             Enable design2 (do halo exhange on few convs) for spatial conv. (default: False)
  --fused-layers FUSED_LAYERS
                        When D2 design is enables for halo exchange, number of blocks to fuse in ResNet model (default: 1)
  --local-DP LOCAL_DP   LBANN intergration of SP with MP. MP can apply data parallelism. 1: only one GPU for a given split, 2: two gpus for a given split (uses DP) (default: 1)
  --slice-method SLICE_METHOD
                        Slice method (square, vertical, and horizontal) in Spatial parallelism (default: square)
  --app APP             Application type (1.medical, 2.cifar, and synthetic) in Spatial parallelism (default: 3)
  --datapath DATAPATH   local Dataset path (default: ./train)
  --enable-master-comm-opt
                        Enable communication optimization for MASTER in Spatial (default: False)
  --enable-evaluation   Enable evaluation mode in GEMS to perform inference (default: False)
  --backend BACKEND     Precision for evaluation [Note: not tested on training] (default: mpi)
  --precision PRECISION
                        Precision for evaluation [Note: not tested on training] (default: fp32)
  --num-workers NUM_WORKERS
                        Slice method (square, vertical, and horizontal) in Spatial parallelism (default: 0)
  --optimizer OPTIMIZER
                        Optimizer (default: adam)
  --learning-rate LEARNING_RATE
                        Learning Rate (default: 0.001)
  --weight-decay WEIGHT_DECAY
                        Weight Decay (default: 0.0001)
  --learning-rate-decay LEARNING_RATE_DECAY
                        Learning Rate Decay (default: 0.85)
  --checkpoint CHECKPOINT
                        Checkpoint path (default: None)