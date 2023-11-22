#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>

#define VERSION 2.2
#define LAST_WORKING_VERSION 2.1

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 16
#define GENERATED_INPUT_HEAD 16
#define THREADS_NUM INITIAL_ARRAY_SIZE

__device__ unsigned int calcSelfGlobalIndex() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ void merge(int globalThreadId, int offset, int* sharedData, int* output){
    int endIdx = globalThreadId + offset*2;
    int middleIdx = globalThreadId + offset;

    int firstHalfIdxCursor = globalThreadId;
    int secondHalfIdxCursor = middleIdx;

    for (unsigned int ptr = globalThreadId; ptr < endIdx; ptr++) {
        if (firstHalfIdxCursor < middleIdx && (secondHalfIdxCursor >= endIdx || sharedData[firstHalfIdxCursor] <= sharedData[secondHalfIdxCursor])) {
            output[ptr] = sharedData[firstHalfIdxCursor];
            firstHalfIdxCursor++;
        } else {
            output[ptr] = sharedData[secondHalfIdxCursor];
            secondHalfIdxCursor++;
        }
    }
}


__global__ void mergeSortGPUBasic(int* input, int* output, int size) {
    extern __shared__ int sharedData[];  // shared memory declaration
    unsigned int localThreadId = threadIdx.x;
    unsigned int globalThreadId = calcSelfGlobalIndex();

    sharedData[localThreadId] = input[globalThreadId];
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        if (localThreadId % (2 * offset) == 0) {
            merge(globalThreadId, offset, sharedData, output);
        }
        __syncthreads();
        sharedData[localThreadId] = output[globalThreadId];
        __syncthreads();
    }

    // Copy the result back to the output array
    output[globalThreadId] = sharedData[localThreadId];
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

void mergesort(int* input, int size){
    int *inputData, *outputData;

    // Allocate memory on GPU
    cudaMalloc(&inputData, size * sizeof(int));
    cudaMalloc(&outputData, size * sizeof(int));

    // Copy the input data to the device
    cudaMemcpy(inputData, input, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocksDim(1, 1, 1);
    dim3 threadBlockDim(THREADS_NUM, 1, 1);

    mergeSortGPUBasic<<<blocksDim,threadBlockDim>>>(inputData, outputData, size);
    cudaDeviceSynchronize(); // wait on CPU side for operations ordered to GPU

    cudaMemcpy(input, outputData, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(inputData);
    cudaFree(outputData);

}

int main() {
    int generatedInputHead = GENERATED_INPUT_HEAD;
    int initialArraySize = INITIAL_ARRAY_SIZE;

    int* randomNumbers = generateRandomInput(initialArraySize);

    // Print the input array
    for (int i = 0; i < generatedInputHead; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    mergesort(randomNumbers, initialArraySize);
   
    // Print the input array
    for (int i = 0; i < initialArraySize; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    delete[] randomNumbers;

    return 0;
}