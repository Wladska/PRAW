#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <cmath>

#define VERSION "2.3"
#define LAST_WORKING_VERSION 2.2

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 20
#define GENERATED_INPUT_HEAD INITIAL_ARRAY_SIZE
#define THREADS_NUM INITIAL_ARRAY_SIZE

__device__ unsigned int calcSelfGlobalIndex() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ void merge(int startIdx, int middleIdx, int endIdx, int* sharedData, int* output){
    int firstHalfIdxCursor = startIdx;
    int secondHalfIdxCursor = middleIdx;

    for (unsigned int ptr = startIdx; ptr <= endIdx; ptr++) {
        if (firstHalfIdxCursor < middleIdx && (secondHalfIdxCursor >= endIdx || sharedData[firstHalfIdxCursor] <= sharedData[secondHalfIdxCursor])) {
            output[ptr] = sharedData[firstHalfIdxCursor];
            firstHalfIdxCursor++;
        } else {
            output[ptr] = sharedData[secondHalfIdxCursor];
            secondHalfIdxCursor++;
        }
    }
}

__device__ int calcEndIdx(int cycle, int size){
    int endIdx = threadIdx.x + pow(2, cycle) - 1;
    return endIdx > size? size : endIdx;
}

__device__ int calcMidIdx(int startIdx, int endIdx){
    return startIdx + ceil((endIdx - startIdx)/2);
}

__device__ bool threadTakesPartInCycle(int cycle, int recursionDepth, int size){
    return threadIdx.x % pow(2, cycle) == 0;
}

__global__ void mergeSortGPUBasic(int* input, int* output, int size, int recursionDepth) {
    extern __shared__ int sharedData[];  // shared memory declaration
    unsigned int localThreadId = threadIdx.x;
    unsigned int globalThreadId = calcSelfGlobalIndex();

    sharedData[localThreadId] = input[globalThreadId];
    __syncthreads();

    for (unsigned int cycle = 1; cycle <= recursionDepth; cycle++) {
        if (threadTakesPartInCycle(cycle, recursionDepth, size)) {
            int endIdx = calcEndIdx(cycle, size);
            int middleIdx = calcMidIdx(localThreadId, endIdx);
            merge(localThreadId, middleIdx, endIdx, sharedData, output);
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

int calcRecursionDepthAnd(int size){
    if (size <= 1) {
        return 0;  // Already sorted
    }

    int mid = size / 2;
    int leftDepth = calcRecursionDepth(mid);
    int rightDepth = calcRecursionDepth(size - mid);

    return std::max(left_depth, right_depth) + 1
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

    mergeSortGPUBasic<<<blocksDim,threadBlockDim>>>(inputData, outputData, size, calcRecursionDepth());
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