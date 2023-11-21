#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <random>

#define VARSION 1.0

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 20
#define THREADS_NUM INITIAL_ARRAY_SIZE

__device__ unsigned int calcSelfGlobalIndex() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ void copy(int* source, int* destination, int startIndex, int numberOfElementsToCopy) {
    for (int i = 0; i < numberOfElementsToCopy; i++) {
        destination[startIndex + i] = source[startIndex + i];
    }
}

__device__ void merge(int startIdx, int endIdx, int* inputData, int* outputData) {
    int middleIdx = (startIdx + endIdx) / 2;
    int firstHalfIdxCursor = startIdx;
    int secondHalfIdxCursor = middleIdx;

    for (unsigned int ptr = startIdx; ptr < endIdx; ptr++) {
        if (firstHalfIdxCursor < middleIdx && (secondHalfIdxCursor >= endIdx || inputData[firstHalfIdxCursor] < inputData[secondHalfIdxCursor])) {
            outputData[ptr] = inputData[firstHalfIdxCursor];
            firstHalfIdxCursor++;
        } else {
            outputData[ptr] = inputData[secondHalfIdxCursor];
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

    /*for (unsigned int offset = 1; offset < size; offset *= 2) {
        if (localThreadId % (2 * offset) == 0) {
            merge(localThreadId, localThreadId + offset, sharedData, output);
            copy(output, sharedData, localThreadId, offset);
        }
        __syncthreads();
    }*/

    // Copy the result back to the output array
    output[globalThreadId] = sharedData[localThreadId] + 1;
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

int main() {
    int generatedInputHead = INITIAL_ARRAY_SIZE;
    int initialArraySize = INITIAL_ARRAY_SIZE;

    int* randomNumbers = generateRandomInput(initialArraySize);

    // Print the input array
    for (int i = 0; i < generatedInputHead; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    int *inputData, *outputData;

    // Allocate memory on GPU
    cudaMalloc(&inputData, initialArraySize * sizeof(int));
    cudaMalloc(&outputData, initialArraySize * sizeof(int));

    // Copy the input data to the device
    cudaMemcpy(inputData, randomNumbers, initialArraySize * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocksDim(1, 1, 1);
    dim3 threadBlockDim(THREADS_NUM, 1, 1);

    mergeSortGPUBasic<<<blocksDim,threadBlockDim>>>(inputData, outputData, initialArraySize);
    cudaDeviceSynchronize(); // wait on CPU side for operations ordered to GPU

    int result[initialArraySize];
    cudaMemcpy(result, outputData, initialArraySize * sizeof(int), cudaMemcpyDeviceToHost);

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