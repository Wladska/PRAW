#include <iostream>
#include <cuda_runtime.h>



int main() {
    const int size = 8;
    int arr[size] = {4, 2, 7, 1, 9, 3, 5, 6};

    // Print the sorted array
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}