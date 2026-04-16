# HighPerformanceFinal

## Division of Labor
Landon: Implemented the serial and shared memory CPU implementations. Created the github repo and discord. Created all the python utils to visualize, validate, and trim the dataset. Wrote the README and organized the project structure including helpers.cpp, kmeans_implementations files, and kmeans.cpp.

Kevin: Implemented the cuda GPU and distributed GPU implementations, updated the readme accordingly. Analyzes the graphs.

Brady: Implemented the distributed CPU implementation and updated the readme accordingly. Brady, also performed the scaling studies. Create the graphs. 

## How To Run

Because the centroids in the kmeans algorithm are randomly initialized, each implementation is run
from a single master file that shares the starting centroids between implementations. This allows the 
use of a python validation script to ensure consistent output across implementations.

### Program Arguments

The program takes three arguments:
1. \<k> : The number of clusters.
2. \<epochs> : The number of epochs. 
3. \<input_file> : The path to the trimmed input CSV file containing the data.
4. \<output_dir> : The directory where the output files will be saved (no trailing slash).

Optional Flags:
- `--skip_serial` : Skip executing the serial implementation.
- `--shared_cpu` : Run the shared memory CPU implementation. This implementation requires an argument after the flag to specify the number of threads to use.
- `--cuda_gpu` : Run the CUDA GPU implementation.
- `--dist_cpu` : Run the distributed computing CPU implementation. This implementation requires an argument after the flag to specify the number of threads per cuda GPU block to use.
- `--dist_gpu` : Run the distributed computing GPU implementation. This implementation requires an argument after the flag to specify the number of threads per cuda GPU block to use.

Using these flags, every implementation can be run sequentially or one at a time. Each will report the total execution time and create a unique output file appended with the implementation name e.g. `serial_output.csv`.

### Running on CHPC
Load the necessary modules

` module load gcc cuda intel-mpi cmake python`

When running in the CHPC, if running a GPU implementation ensure a GPU has been allocated, and compile with the following 
` nvcc -ccbin mpicxx -Xcompiler -fopenmp ./kmeans_implementations/*.cpp ./kmeans_implementations/*.cu -o kmeans`

If running an implementation that uses MPI this command is required for compatibility
`export I_MPI_FABRICS=shm`

The command to run all implementations sequentially on the CHPC is the following:
```bash
mpirun -n <num_nodes> ./kmeans <k> <epochs> ./csvs/trimmed_track_features.csv ./csvs --shared_cpu <num_threads> --cuda_gpu <threads_in_block_cuda> --dist_cpu --dist_gpu <threads_in_block_dist>
```

*** NOTE *** : this method of running is not ideal and may harm the performance of each method. For scaling studies and best performance it is recommended to run one method at a time. In the following sections commands are given to run each implementation stand-alone. Compilation is the same for every method and that command can be seen above. Also, keep in mind that when running all the implementations simultaneously the <num_threads> arg for shared_cpu will be multiplied by the num_nodes argument. 

**Running all implementations simultaneously should only be used to validate output in combination with the python validation script.**

*** NOTE *** : If running with the `mpirun` command, ensure one of the "dist" implementations is being used, using `mpirun` without including a dist implementation exlcudes the 
logic that handles each rank and will result in all the non-dist implementations being run `num_nodes` times. Following the instructions below to run each implementation one at a time
will prevent this issue. 

### Serial Implementation

Example execution of serial implementation: 

```bash
./kmeans 4 25 ./csvs/trimmed_track_features.csv ./csvs
```

### Shared Memory CPU Implementation

Example execution of shared memory CPU implementation:

```bash
./kmeans 4 25 ./csvs/trimmed_track_features.csv ./csvs --shared_cpu 8 --skip_serial
```

### CUDA GPU Implementation

Example execution of CUDA GPU implementation:

```bash
./kmeans 4 25 ./csvs/trimmed_track_features.csv ./csvs --cuda_gpu 256 --skip_serial
```

### Distributed CPU Implementation

Example execution of Distributed CPU implementation:

```bash
mpirun -n 2 ./kmeans 4 25 ./csvs/trimmed_track_features.csv ./csvs --dist_cpu --skip_serial
```

### Distributed GPU Implementation

Example execution of Distributed GPU implementation:

```bash
./kmeans 4 25 ./csvs/trimmed_track_features.csv ./csvs --dist_gpu 256 --skip_serial
```

## Python Utilities

It is recommended to create a python virtual environment to run the python utilities. A requirements.txt file has been included to easily install dependencies. To create a virtual environment and install dependencies, run the following commands:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

### Trim Dataset
The `trim_dataset.py` script is used to trim the original dataset down to 3 attributes to be used for the kmeans implementations. By default we use energy, speechiness, and liveness. The script takes the following arguments:
1. \<input_file> : Path to the input CSV file containing the full dataset.
2. \<output_file> : Path to the output CSV file for the trimmed dataset.

Example execution of trim dataset script:

```bash
python trim_dataset.py ./csvs/track_features.csv ./csvs/trimmed_track_features.csv
```

Note: As the full dataset is too large to be included in the repo, an already trimmed version has been included. The full dataset can be found at https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

### Visualize Output
The `visualize.py` script is used to visualize the output of the kmeans implementations. It takes the following arguments:
1. \<input_file> : Path to the input CSV file containing the kmeans output.
2. \<output_file> : Path to create the output image file (including extension).

Example execution of visualize script:

```bash
python visualize.py ./csvs/output_serial.csv ./images/serial.png
```

Example output image from visualize script:
![Example Visualization](./images/sample.png)

### Validation Script
The `validateResults.py` script is used to validate the results of the kmeans implementations by comparing multiple output files to each other to ensure they are consistent. It takes the following arguments:

1. \<input_file1> : Path to the first input CSV file containing the kmeans output.
2. \<input_file2> : Path to the second input CSV file containing the kmeans output.

Example execution of validate results script:

```bash
python validateResults.py ./csvs/output_serial.csv ./csvs/output_shared_cpu.csv
```

### Scaling Study
For the scaling study, we chose k={2,4,6}, e={25,50,100}. We chose these k and e values to give us a clear understanding of how these values affect the k-mean clustering's runtime. The problem sizes we chose were 2^20, 2^19, and 2^18.  
For the parallel vs serial scaling study, we used eight threads for the parallel implementation. Since there is only one thread on the serial, we run the program eight times and we include the speedup and efficiency. 
| Parallel | Serial |
|--------|----------|
|K2 E25|
|![Graph](./graphs/K2_E25_Parallel.png)| ![Graph](./graphs/K2_E25_Serial.png)|
|K2 E50|
|![Graph](./graphs/K2_E50_ParallelScaling.png)| ![Graph](./graphs/K2_E50_SerialScaling.png)|
|K2 E100|
|![Graph](./graphs/K2_E100_ParallelScaling.png)| ![Graph](./graphs/K2_E100_SerialScaling.png)|
|K4 E25|
|![Graph](./graphs/K4_E25_ParallelScaling.png)|![Graph](./graphs/K4_E25_SerialScaling.png)|
|K4 E50|
|![Graph](./graphs/ParallelCPUK4E50Scaling.png)|![Graph](./graphs/SerialK4E50.png)|
|K4 E100|
|![Graph](./graphs/ParallelCPUK4E100Scaling.png)|![Graph](./graphs/SerialK4E100.png)|
|K6 E25|
|![Graph](./graphs/ParallelCPUK6E25Scaling.png)|![Graph](./graphs/SerialK6E25.png)|
|K6 E50|
|![Graph](./graphs/ParallelCPUK6E50Scaling.png)|![Graph](./graphs/SerialK6E50.png)|
|K6 E100|
|![Graph](./graphs/ParallelCPUK6E100Scaling.png)|![Graph](./graphs/SerialK6E100.png)|
|Parallel Speed up and Efficiency|
|K2 E25|
|![Graph](./graphs/K2_E25_ParallelSP.png)|![Graph](./graphs/K2_E25_ParallelE.png)|
|K2 E50|
|![Graph](./graphs/K2_E50_ParallelSP.png)|![Graph](./graphs/K2_E50_ParallelE.png)|
|K2 E100|
|![Graph](./graphs/K2_E100_ParallelSpeedup.png)|![Graph](./graphs/K2_E100_ParallelE.png)|
|K4 E25|
|![Graph](./graphs/K4_E25_ParallelSpeedup.png)|![Graph](./graphs/K4_E25_ParallelE.png)|
|K4 E50|
|![Graph](./graphs/ParallelCPUK4E50Speedup.png)|![Graph](./graphs/ParallelCPUK4E50Efficiency.png)|
|K4 E100|
|![Graph](./graphs/ParallelCPUK4E100Speedup.png)|![Graph](./graphs/ParallelCPUK4E100Efficiency.png)|
|K6 E25|
|![Graph](./graphs/ParallelCPUK6E25Speedup.png)|![Graph](./graphs/ParallelCPUK6E25Efficiency.png)|
|K6 E50|
|![Graph](./graphs/ParallelCPUK6E50Speedup.png)|![Graph](./graphs/ParallelCPUK6E50Efficiency.png)|
|K6 E100|
|![Graph](./graphs/ParallelCPUK6E100Speedup.png)|![Graph](./graphs/ParallelCPUK6E100Efficiency.png)|

For the GPU study, we used the blocksize 8x8, 16x16, and 32x32.
| GPU    |
|--------|
|K2 E25|
|![Graph](./graphs/K2_E25_GPUScaling.png)|
|K2 E50|
|![Graph](./graphs/K2_E50_GPUScaling.png)|
|K2 E100|
|![Graph](./graphs/K2_E100_GPUScaling.png)|
|K4 E25|
|![Graph](./graphs/K4_E25_GPUScaling.png)|
|K4 E50|
|![Graph](./graphs/GPUK4E50Scaling.png)|
|K4 E100|
|![Graph](./graphs/GPUK4E100Scaling.png)|
|K6 E25|
|![Graph](./graphs/GPUK6E25Scaling.png)|
|K6 E50|
|![Graph](./graphs/GPUK6E50Scaling.png)|
|K6 E100|
|![Graph](./graphs/GPUK6E100Scaling.png)|

For the Distribute GPU scaling study, we used the same blocksize as the GPU.
We used 2-4 GPU for the Distribute GPU and 2-4 Nodes for the Distribute CPU.

| Distribute CPU | Distribute GPU |
|--------|----------|
|K2 E25|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K2 E 25 NP 4.png>)|![Graph](./graphs/K2_E25_DistributeGPUBS64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K2 E 25 NP 6.png>)|![Graph](./graphs/K2_E25_DistributeGPUBS256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K2 E 25 NP 8.png>)|![Graph](./graphs/K2_E25_DistributeGPUBS1024.png) |
|K2 E50|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K2 E 50 NP 4.png>)|![Graph](./graphs/K2_E50_DistributeGPUBS64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K2 E 50 NP 6.png>)|![Graph](./graphs/K2_E50_DistributeGPUBS256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K2 E 50 NP 8.png>)|![Graph](./graphs/K2_E50_DistributeGPUBS1024.png) |
|K2 E100|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K2 E 100 NP 4.png>)|![Graph](./graphs/K2_E100_DistributeGPUBS64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K2 E 100 NP 6.png>)|![Graph](./graphs/K2_E100_DistributeGPUBS256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K2 E 100 NP 8.png>)|![Graph](./graphs/K2_E100_DistributeGPUBS1024.png) |
|K4 E25|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K4 E 25 NP 4.png>)|![Graph](./graphs/K4_E25_DistributeGPUBS64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K4 E 25 NP 6.png>)|![Graph](./graphs/K4_E25_DistributeGPUBS256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K4 E 25 NP 8.png>)|![Graph](./graphs/K4_E25_DistributeGPUBS1024.png) |
|K4 E50|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K4 E 50 NP 4.png>)|![Graph](./graphs/DistributeGPUK4E50Blocksize64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K4 E 50 NP 6.png>)|![Graph](./graphs/DistributeGPUK4E50Blocksize256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K4 E 50 NP 8.png>)|![Graph](./graphs/DistributeGPUK4E50Blocksize1024.png) |
|K4 E100|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K4 E 100 NP 4.png>)|![Graph](./graphs/DistributeGPUK4E100Blocksize64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K4 E 100 NP 6.png>)|![Graph](./graphs/DistributeGPUK4E100Blocksize256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K4 E 100 NP 8.png>)|![Graph](./graphs/DistributeGPUK4E100Blocksize1024.png) |
|K6 E25|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K6 E 25 NP 4.png>)|![Graph](./graphs/DistributeGPUK6E25Blocksize64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K6 E 25 NP 6.png>)|![Graph](./graphs/DistributeGPUK6E25Blocksize256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K6 E 25 NP 8.png>)|![Graph](./graphs/DistributeGPUK6E25Blocksize1024.png) |
|K6 E50|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K6 E 50 NP 4.png>)|![Graph](./graphs/DistributeGPUK6E50Blocksize64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K6 E 50 NP 6.png>)|![Graph](./graphs/DistributeGPUK6E50Blocksize256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K6 E 50 NP 8.png>)|![Graph](./graphs/DistributeGPUK6E50Blocksize1024.png) |
|K6 E100|
| Number of proceses (NP) 4|Blocksize 64|
|![Graph](<./graphs/Distribute CPU K6 E 100 NP 4.png>)|![Graph](./graphs/DistributeGPUK6E100Blocksize64.png) 
| NP 6|Blocksize 256|
|![Graph](<./graphs/Distribute CPU K6 E 100 NP 6.png>)|![Graph](./graphs/DistributeGPUK6E100Blocksize256.png) |
| NP 8|Blocksize 1024|
|![Graph](<./graphs/Distribute CPU K6 E 100 NP 8.png>)|![Graph](./graphs/DistributeGPUK6E100Blocksize1024.png) |