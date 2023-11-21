#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 20
#define HEAD_PRINT_INPUT_DATA 20

/// only the x axis y and z are set to 1
#define KERNELS_NUM 1 // unused
#define BLOCKS_NUM 1 // unused
#define THREADS_NUM INITIAL_ARRAY_SIZE

__device__ unsigned int calcSelfGlobalIndex(){
    return threadIdx.x + blockIdx.x * blockDim.x;
}


__device__ copy(int* source, int* destination, int startIndex, int numberOfElementsToCopy) {
    for(int i = 0; i < numberOfElementsToCopy; i++) {
        destination[startIndex + i] = source[startIndex + i];
    }
}

/*
Input:          8 3 1 9 1 2 7 5 9 3 6 4 2 0 2 5
Threads: |    t1    |    t2    |    t3    |    t4    |
         | 8 3 1 9  | 1 2 7 5  | 9 3 6 4  | 2 0 2 5  |
         |  38 19   |  12 57   |  39 46   |  02 25   |
         |   1398   |   1257   |   3469   |   0225   |
         +----------+----------+----------+----------+
         |          t1         |          t3         |
         |       11235789      |       02234569      |
         +---------------------+---------------------+
         |                     t1                    |
         |      0 1 1 2 2 2 3 3 4 5 5 6 7 8 9 9      |

*/

__device__ merge(int startIdx, int endIdx, int* inputData, int* outputData) {
    int middleIdx = (startIdx + endIdx) / 2;
    int firstHalfIdxCursor = startIdx;
    int secondHalfIdxCursor = middleIdx;

    for (unsigned int ptr = startIdx; ptr < startIdx + offset, ptr ++) {
        if (firstHalfIdxCursor < middleIdx && (secondHalfIdxCursor >= endIdx || inputData[firstHalfIdxCursor] < inputData[secondHalfIdxCursor]))
        {
            outputData[ptr] = inputData[firstHalfIdxCursor];
            firstHalfIdxCursor ++;
        } else {
            outputData[ptr] = inputData[secondHalfIdxCursor];
            secondHalfIdxCursor ++;
        }
    }
}

__global__ void mergeSortGPUBasic(int* input, int* midMerge, int size) {
    extern __shared__ int sharedData []; // shared memory declaration
    unsigned int localThreadId = threadIdx.x;
    unsigned int globalThreadId = calcSelfGlobalIndex();
    sharedData[localThreadId] = input[globalThreadId];
    __syncthreads () ;
    
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        if ( localThreadId % (2* offset) == 0) {
            merge(localThreadId, localThreadId + offset, sharedData, midMerge);
            copy(midMerge, sharedData, localThreadId, offset)
        }
        __syncthreads () ;
    }
}

int* generateRandomInput(int size) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<int> distribution(MIN_DISTRIBUTION, MAX_DISTRIBUTION);

    int* randomNumbers = new int[size];

    for (int i = 0; i < size; ++i) {
        randomNumbers[i] = distribution(generator);
    }

    return randomNumbers;
}

int main(int argc, char *argv[]) {
    std::cout << "Usage: " << argv[0] << " <integer_argument> <generated_input_head_number>" << std::endl;

    int generatedInputHead = HEAD_PRINT_INPUT_DATA;
    int initialArraySize = INITIAL_ARRAY_SIZE;

    if (argc >= 2) {
        initialArraySize = std::atoi(argv[1]);
        if (argc > 2) {
            generatedInputHead = std::atoi(argv[2]);
            if (generatedInputHead > initialArraySize) {
                generatedInputHead = initialArraySize
            }
        }
    }

    int* randomNumbers = generateRandomInput(initialArraySize);
    int *inputData, *midMergeData;

    // Allocate memory on GPU
    cudaMalloc(&inputData, initialArraySize * sizeof(int));
    cudaMalloc(&midMergeData, initialArraySize * sizeof(int));

    // Copy the input data to the device
    cudaMemcpy(inputData, randomNumbers, initialArraySize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocksDim(1,1,1);
    dim3 threadBlockDim(THREADS_NUM,1,1);

    mergeSortGPUBasic<blocksDim,threadBlockDim>(inputData, midMergeData, initialArraySize);
    cudaDeviceSynchronize(); // wait on CPU side for operations ordered to GPU

    int result[initialArraySize];
    cudaMemcpy(result, d_input, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the input array
    for (int i = 0; i < generatedInputHead; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    // Print the sorted array
    for (int i = 0; i < initialArraySize; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    delete[] randomNumbers;

    // Free allocated memory on the device
    cudaFree(inputData);
    cudaFree(outputData);

    return 0;
}