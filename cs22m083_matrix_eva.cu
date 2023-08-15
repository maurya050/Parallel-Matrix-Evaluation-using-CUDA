#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

// kernel for matrix multiplication, C = AB
__global__ void matrixMul(int *A, int *B, int *C, int p, int q, int r, int n)
{
	
    int x1 = blockIdx.x * blockDim.y + threadIdx.x;
    int y1 =  blockIdx.y * blockDim.x + threadIdx.y;
    __shared__ int s_a[32][32];
    __shared__ int s_b[32][32];
    int i, z, sum = 0, nMul = p, MulInBlock = 32, a_y1, a_x1, b_y1, b_x1;
    
    for (z = 0; z < n; z++)
    {

        // Load Matrix A
        a_y1 = (blockIdx.y * blockDim.x + threadIdx.y);
		int a1 = p * a_y1;
        a_x1 = (z * blockDim.x + threadIdx.x);
		a1 = a1 + a_x1;
        if (a_y1 < q && a_x1 < p)
        {
            s_a[threadIdx.y][threadIdx.x] = A[a1];
        }

        // Load Matrix B
        b_y1 = (z * blockDim.x + threadIdx.y);
		int b1 = r * b_y1;
        b_x1 = (blockIdx.x * blockDim.x + threadIdx.x);
		b1 = b1 + b_x1;
        if (b_y1 < p && b_x1 < r)
        {
            s_b[threadIdx.y][threadIdx.x] = B[b1];
        }

        __syncthreads();

        if (x1 < r && y1 < q)
        {
            if (nMul < blockDim.x)
            {
                MulInBlock = nMul;
            }
            int ix = threadIdx.x;
            int iy = threadIdx.y;
            for (i = 0; i < MulInBlock; i++)
            {
                sum += s_a[iy][i] * s_b[i][ix];
            }

            nMul -= blockDim.x;
        }

        __syncthreads();
    }
	int asgn = y1 * r + x1;
    /* 
     * In the Matrix C, checking to see if that thread actually belongs to a valid position.
     */
    if (x1 < r && y1 < q)
    {
        C[asgn] = sum;
    }
}

// kernel for transpose
__global__ void transpose(int r, int q, int *input, int *output) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	int i, j;
    __shared__ int temp[32][32+1]; // Use padding for coalesced accesses
    if (x < q and y < r)
    {
		int in_idx = y * q + x;
		i = threadIdx.x;
		j = threadIdx.y;
        // Load a 32x32 block of elements into shared memory
                temp[j][i] = input[in_idx]; 
        // Transpose the block
	}
	__syncthreads();
        
    int out_index = blockIdx.y * blockDim.x + threadIdx.x;
    int in_index =  blockIdx.x* blockDim.y + threadIdx.y;
    int out = (in_index * r ) + out_index;
	i = threadIdx.x;
	j = threadIdx.y;
	if( out_index < r and in_index < q)
	{
        output[out] = temp[i][j];
	}
}

// kernel for matrix addition, A = A+B
__global__ void add_matrices(int *matrixA, int *matrixB) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    matrixA[id] =  matrixA[id] + matrixB[id];
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));
 
	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	int *d_transD,*d_matrixTemp;
	int X = (q+ (32-1))/32;
	int Y = (r+ (32-1))/32;
	dim3 grid1(X, Y);
	dim3 block(32, 32);
    cudaMalloc(&d_transD, r * q * sizeof(int));//Calculating transpose of matrix D
	transpose<<<grid1, block>>>(r, q, d_matrixD, d_transD);
    cudaFree(d_matrixD); d_matrixD = d_transD;

	/* Write your code here */
	/* Configure and launch kernels */

	/* ****************************************************************** */
    cudaMalloc(&d_matrixE, p * r * sizeof(int));
	int nBlocks1 = (int) std::ceil( (double) q / 32);
	int X1 = (r+ (32-1))/32;
	int Y1 = (p+ (32-1))/32;
	dim3 grid2(X1, Y1);
    matrixMul<<< grid2, block>>>( d_matrixA, d_matrixB, d_matrixE,q, p,r, nBlocks1);

	int X2 = (r+ (32-1))/32;
	int Y2 = (p+ (32-1))/32;
	dim3 grid3(X2, Y2);
	cudaMalloc(&d_matrixTemp, p * r * sizeof(int));
	int nBlocks2 = (int) std::ceil( (double) q / 32);
	matrixMul<<<grid3, block>>>( d_matrixC, d_matrixD,d_matrixTemp, q, p, r, nBlocks2);
	add_matrices<<<p, r>>>(d_matrixE, d_matrixTemp);
    cudaFree(d_matrixTemp);
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
	
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
