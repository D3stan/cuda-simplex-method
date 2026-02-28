#include <iostream>

__global__ void helloCUDA() {
    printf("Hello World dalla GPU! Sono il Thread %d all'interno del Blocco %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    std::cout << "Avvio da CPU..." << std::endl;

    // Lanciamo il kernel
    helloCUDA<<<2, 4>>>();

    // 1. Controlliamo se ci sono stati errori nel LANCIO del kernel
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Errore di lancio del Kernel: " << cudaGetErrorString(launchErr) << std::endl;
    }

    // 2. Sincronizziamo e controlliamo se ci sono stati errori durante l'ESECUZIONE
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "Errore di esecuzione o sincronizzazione CUDA: " << cudaGetErrorString(syncErr) << std::endl;
    }

    std::cout << "Esecuzione GPU terminata." << std::endl;
    return 0;
}