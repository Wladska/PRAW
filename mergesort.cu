#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>
#include <cmath>

#define VERSION "2.6"
#define LAST_WORKING_VERSION 2.2

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 10
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
    int endIdx = threadIdx.x + powf(2, cycle) - 1;
    return endIdx > size? size : endIdx;
}

__device__ int calcMidIdx(int startIdx, int endIdx){
    return startIdx + ceilf((endIdx - startIdx)/2.0);
}

__device__ bool threadTakesPartInCycle(int cycle, int localThreadId){
    int power = powf(2, cycle);
    return localThreadId % power == 0;
}

__global__ void mergeSortGPUBasic(int* input, int* output, int size, int recursionDepth) {
    extern __shared__ int sharedData[];  // shared memory declaration
    unsigned int localThreadId = threadIdx.x;
    unsigned int globalThreadId = calcSelfGlobalIndex();

    sharedData[localThreadId] = input[globalThreadId];
    __syncthreads();

    for (unsigned int cycle = 1; cycle <= recursionDepth; cycle++) {
        if (threadTakesPartInCycle(cycle, localThreadId)) {
            int endIdx = calcEndIdx(cycle, size);
            output[globalThreadId] = endIdx;
            int middleIdx = calcMidIdx(localThreadId, endIdx);
            output[globalThreadId] = middleIdx;
            merge(localThreadId, middleIdx, endIdx, sharedData, output);
        }
        __syncthreads();
        sharedData[localThreadId] = output[globalThreadId];
        __syncthreads();
    }

    // Copy the result back to the output array
    //output[globalThreadId] = sharedData[localThreadId];
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

int calcRecursionDepth(int size){
    if (size <= 1) {
        return 0;  // Already sorted
    }

    int mid = size / 2;
    int leftDepth = calcRecursionDepth(mid);
    int rightDepth = calcRecursionDepth(size - mid);

    return std::max(leftDepth, rightDepth) + 1;
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

    int recDepth = calcRecursionDepth(size);

    mergeSortGPUBasic<<<blocksDim,threadBlockDim>>>(inputData, outputData, size, recDepth);
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
   
    // Print the sorted array
    for (int i = 0; i < initialArraySize; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    delete[] randomNumbers;

    return 0;
}