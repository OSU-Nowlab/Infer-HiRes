# Run Single GPU Inference:

#### Generic command:
```bash

python single_gpu_inference/benchmark_resnet_inference.py \
        --batch-size ${batch_size} \
        --image-size ${image_size} \
        --precision ${precision} \
        --checkpoint ${checkpoint} \
        --datapath ${datapath} \
        --enable-evaluation

```
#### Examples

- With 4 GPUs [split size: 4]

Example to run ResNet for 1024 * 1024 image size woth INT8 quantization.

```bash

python single_gpu_inference/benchmark_resnet_inference.py \
        --batch-size 1 \
        --image-size 1024 \
        --precision 'int8' \
        --enable-evaluation
```



Below are the available configuration options :

<pre>
optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Prints performance numbers or logs (default: False)
  --batch-size BATCH_SIZE
                        input batch size (default: 32)
  --image-size IMAGE_SIZE
                        Image size for synthetic benchmark (default: 32)
  --num-classes NUM_CLASSES
                        Number of classes (default: 10)
  --app APP             Application type (1.medical, 2.cifar, and synthetic) in Spatial parallelism (default: 3)
  --datapath DATAPATH   local Dataset path (default: ./train)
  --enable-evaluation   Enable evaluation mode in GEMS to perform inference (default: False)
  --backend BACKEND     Precision for evaluation [Note: not tested on training] (default: mpi)
  --precision PRECISION
                        Precision for evaluation [Note: not tested on training] (default: fp32)
  --num-workers NUM_WORKERS
                        Slice method (square, vertical, and horizontal) in Spatial parallelism (default: 0)
  --checkpoint CHECKPOINT
                        Checkpoint path (default: None)