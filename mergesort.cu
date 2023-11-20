#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>

#define MIN_DISTRIBUTION -10000
#define MAX_DISTRIBUTION 10000

#define INITIAL_ARRAY_SIZE 10
#define HEAD_PRINT_INPUT_DATA 10

std::vector<int> generateRandomInput(int size) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_int_distribution<int> distribution(MIN_DISTRIBUTION, MAX_DISTRIBUTION);

    std::vector<int> randomNumbers;
    randomNumbers.reserve(size);

    for (int i = 0; i < numElements; ++i) {
        int randomNum = distribution(generator);
        randomNumbers.push_back(randomNum);
    }

    return randomNumbers;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <integer_argument> <generated_input_head_number>" << std::endl;
        return 1;
    }

    int generatedInputHead = HEAD_PRINT_INPUT_DATA;
    int initialArraySize = INITIAL_ARRAY_SIZE;

    try {
        initialArraySize = std::atoi(argv[1]);
        if (argc > 2) {
            generatedInputHead = std::atoi(argv[2]);
        }
    } catch {
        std::cerr << "ArgumentS were not a valid integer" << std::endl;
        return 1;
    }

    td::vector<int> randomNumbers = generateRandomInput(initialArraySize);

    // Print the sorted array
    for (int i = 0; i<generatedInputHead; i++) {
        std::cout << randomNumbers[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}