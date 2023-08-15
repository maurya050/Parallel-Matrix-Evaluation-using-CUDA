# Parallel-Matrix-Evaluation-using-CUDA
Parallel Matrix Evaluation using CUDA by considering the aspects of memory coalescing and shared memory.


#### Goal:
For the 4 different Matices A, B, C, and D of dimension p×q, q×r, p×q, r×q respectively. Computing another Matrix E after performing the below-given equation on the matrices. - *by considering the aspects of memory coalescing and shared memory.*


$$
E = AB + CD^T
$$


* Implementing memory coalescing to ensure contiguous memory accesses, optimizing memory throughput, and maximizing
GPU utilization during matrix calculations

* Leveraged shared memory effectively to minimize global memory accesses and accelerate matrix computation tasks.
