#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <chrono>

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 10000
#define RUNS 10
#define DEBUG 0

std::vector<int> generateRandomInput(int size) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::default_random_engine generator(std::rand());
    std::uniform_int_distribution<int> distribution(MIN_DISTRIBUTION, MAX_DISTRIBUTION);

    std::vector<int> randomNumbers{};

    for (int i = 0; i < size; ++i) {
        randomNumbers.push_back(distribution(generator));
    }

    return randomNumbers;
}

void merge(std::vector<int>& array, int const left, int const mid, int const right) {
    std::vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        if (array[i] <= array[j]) {
            temp[k++] = array[i++];
        } else {
            temp[k++] = array[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = array[i++];
    }

    while (j <= right) {
        temp[k++] = array[j++];
    }

    for (i = left; i <= right; i++) {
        array[i] = temp[i - left];
    }
}

void mergeSort(std::vector<int>& array, int begin, int end) {
    if (begin >= end) {
        return;
    }

    int mid = begin + (end - begin) / 2;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mergeSort(array, begin, mid);
        }

        #pragma omp section
        {
            mergeSort(array, mid + 1, end);
        }
    }

    merge(array, begin, mid, end);
}

int main() {
    int initialArraySize = INITIAL_ARRAY_SIZE;
    std::vector<double> runs;
    double wholeDuration = 0;

    for (int i = 0; i < RUNS; i++) {
        std::vector<int> randomNumbers = generateRandomInput(initialArraySize);

        if (DEBUG) {
            std::cout << "Original array: \n";
            for (const auto& el : randomNumbers) {
                std::cout << el << " ";
            }
            std::cout << "\n";
        }

        auto start = std::chrono::high_resolution_clock::now();
        // Perform parallel merge sort
        mergeSort(randomNumbers, 0, initialArraySize - 1);
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        double time = duration.count() / 1000000.0;

        std::cout << "Duration: " << time << " seconds" << std::endl;
        runs.push_back(time);
        wholeDuration += time;

        if (DEBUG) {
            std::cout << "Sorted array: \n";
            for (const auto& el : randomNumbers) {
                std::cout << el << " ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "Mean : " << wholeDuration/(double) RUNS << std::endl;
    std::sort(runs.begin(), runs.end());
    std::cout << "Uncertainty: " << (runs[runs.size() - 1] - runs[0])/2 << std::endl;

    return 0;
}