
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <algorithm>
#include <chrono>

using BASE_TYPE = double;
using uint = unsigned int;

constexpr uint BLOCK_SIZE = 16;

__global__ void matrixMultiplyKernel(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, uint Acols, uint Bcols)
{
	// индекс перемножаемого элемента строки матрицы A
	uint i = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
	// индекс перемножаемого элемента столбца матрицы B
	uint j = blockDim.x * blockIdx.x + threadIdx.x;
	
	BASE_TYPE sum = 0;
	for (uint k = 0; k < Acols; k++)
	{
		sum += A[i + k] * B[k * Bcols + j];
	}

	uint idx = Bcols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;

	C[idx] = sum;
}

void matrixMultiplyCpu(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, dim3 Adim, uint Bcols)
{
	// цикл по строкам матрицы A
	for (uint i = 0; i < Adim.x; i++)
	{
		// индекс элемента в строке A(на какой столбец B умножаем строку)
		for (uint j = 0; j < Bcols; j++)
		{
			BASE_TYPE sum = 0;
			for (uint k = 0; k < Adim.y; k++)
			{
				sum += A[i * Adim.x + k] * B[k * Bcols + j];
			}
			int ind = Bcols * i + j;
			C[ind] = sum;
		}
	}
}

// Функция вычисления числа, которое больше числа a и кратное числу b
uint toMultiple(uint a, uint b)
{
	auto mod = a % b;

	if (mod != 0)
	{
		mod = b - mod;
		return a + mod;
	}

	return a;
}

void PrintMatrixResult(const BASE_TYPE* A, const BASE_TYPE* B, BASE_TYPE* C, dim3 Adim, dim3 Bdim)
{
	// shorten output
	if (Adim.x > 6 || Adim.y > 6 || Bdim.y > 6)
	{
		std::cout << C[0] << " " << C[1] << " " << C[2] << " ... " << C[Bdim.y - 1] << std::endl;
		std::cout << C[Bdim.y] << " " << C[Bdim.y + 1] << " " << C[Bdim.y + 2] << " ... " << C[Bdim.y * 2] << std::endl;
		std::cout << "................" << std::endl;
		std::cout << C[(Adim.x - 1) * Bdim.y] << " " << C[(Adim.x - 1) * Bdim.y + 1] 
			<< " " << C[(Adim.x - 1) * Bdim.y + 2] << "..." << C[(Adim.x - 1) * Bdim.y + (Adim.y - 1)] << std::endl;
	}
	// full output
	else {
		for (size_t i = 0; i < Adim.x; i++)
		{
			for (size_t j = 0; j < Adim.y; j++)
			{
				std::cout << A[i * Adim.y + j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\tX" << std::endl;
		for (size_t i = 0; i < Bdim.x; i++)
		{
			for (size_t j = 0; j < Bdim.y; j++)
			{
				std::cout << B[i * Bdim.y + j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\t=" << std::endl;
		for (size_t i = 0; i < Adim.x; i++)
		{
			for (size_t j = 0; j < Bdim.y; j++)
			{
				std::cout << C[i * Bdim.y + j] << " ";
			}
			std::cout << "\n";
		}
	}
}

void cudaMatrixMultiply(const BASE_TYPE* h_A, const BASE_TYPE* h_B, BASE_TYPE* h_C,
	uint Asize, uint Bsize, uint Csize, dim3 Adim, dim3 Bdim);

int main()
{
	unsigned int Arows, Acols, Brows, Bcols;
	dim3 Adim, Bdim;

	std::cout << "Enter number of rows of A matrix: ";
	std::cin >> Arows;
	std::cout << "Enter number of columns of A matrix: ";
	std::cin >> Acols;

	Brows = Acols;

	std::cout << "Enter number of columns of B matrix: ";
	std::cin >> Bcols;

	Arows = toMultiple(Arows, BLOCK_SIZE);
	std::cout << "Arows = " << Arows << "\n";

	Acols = toMultiple(Acols, BLOCK_SIZE);
	std::cout << "Acols = " << Acols << "\n";

	Brows = toMultiple(Brows, BLOCK_SIZE);
	std::cout << "Brows = " << Brows << "\n";

	Bcols = toMultiple(Bcols, BLOCK_SIZE);
	std::cout << "Bcols = " << Bcols << "\n";

	uint Asize = Arows * Acols;
	uint Bsize = Brows * Bcols;
	uint Csize = Arows * Bcols;

	BASE_TYPE* h_A = new BASE_TYPE[Asize];
	BASE_TYPE* h_B = new BASE_TYPE[Bsize];
	BASE_TYPE* h_C1 = new BASE_TYPE[Csize];
	BASE_TYPE* h_C2 = new BASE_TYPE[Csize];

	auto randNumber = []() {
		return rand() % 10;
			//(BASE_TYPE)RAND_MAX;
	};

	std::generate(h_A, h_A + Asize, randNumber);
	std::generate(h_B, h_B + Bsize, randNumber);

	Adim = { Arows, Acols };
	Bdim = { Brows, Bcols };

	using namespace std::chrono;

	auto start = high_resolution_clock::now();

	matrixMultiplyCpu(h_A, h_B, h_C1, Adim, Bcols);

	auto stop = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(stop - start);

	std::cout << "Time spent executing by the CPU: "
		<< duration.count() << " milliseconds" << std::endl;

	PrintMatrixResult(h_A, h_B, h_C1, Adim, Bdim);

	cudaMatrixMultiply(h_A, h_B, h_C2, Asize, Bsize, Csize, Adim, Bdim);
	
	PrintMatrixResult(h_A, h_B, h_C2, Adim, Bdim);

	return 0;
}

void cudaMatrixMultiply(const BASE_TYPE* h_A, const BASE_TYPE* h_B, BASE_TYPE* h_C,
	uint Asize, uint Bsize, uint Csize, dim3 Adim, dim3 Bdim)
{
	BASE_TYPE* dev_A = 0;
	BASE_TYPE* dev_B = 0;
	BASE_TYPE* dev_C = 0;
	cudaError_t cudaStatus;

	auto checkError = [&](cudaError_t status)
	{
		if (status != cudaSuccess)
		{
			std::cerr << "Error! ";
			std::cerr << cudaGetErrorString(status) << std::endl;
			cudaFree(dev_A);
			cudaFree(dev_B);
			cudaFree(dev_C);
			exit(EXIT_FAILURE);
		}
	};

	cudaStatus = cudaSetDevice(0);
	checkError(cudaStatus);

	cudaStatus = cudaMalloc((void**)&dev_A, Asize * sizeof(BASE_TYPE));
	checkError(cudaStatus);
	cudaStatus = cudaMalloc((void**)&dev_B, Bsize * sizeof(BASE_TYPE));
	checkError(cudaStatus);
	cudaStatus = cudaMalloc((void**)&dev_C, Csize * sizeof(BASE_TYPE));
	checkError(cudaStatus);

	cudaStatus = cudaMemcpy(dev_A, h_A, Asize * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	checkError(cudaStatus);
	cudaStatus = cudaMemcpy(dev_B, h_B, Bsize * sizeof(BASE_TYPE), cudaMemcpyHostToDevice);
	checkError(cudaStatus);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);

	auto Bcols = Bdim.y;
	auto Arows = Adim.x;
	dim3 blocksPerGrid = dim3(Bcols / BLOCK_SIZE, Arows / BLOCK_SIZE);
	
	// инициализируем события
	cudaEvent_t start, stop;
	float elapsedTime;
	// создаем события
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// запись события
	cudaEventRecord(start, 0);

	matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, Adim.y, Bcols);
	
	cudaStatus = cudaEventRecord(stop, 0);
	checkError(cudaStatus);
	cudaStatus = cudaEventSynchronize(stop);
	checkError(cudaStatus);
	cudaStatus = cudaEventElapsedTime(&elapsedTime, start, stop);
	checkError(cudaStatus);
	// вывод информации
	printf("Time spent executing by the GPU: %.2f milliseconds\n", elapsedTime);
	// уничтожение события
	cudaStatus = cudaEventDestroy(start);
	checkError(cudaStatus);
	cudaStatus = cudaEventDestroy(stop);
	checkError(cudaStatus);

	cudaStatus = cudaMemcpy(h_C, dev_C, Csize * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
	checkError(cudaStatus);

	// Free resources.
	cudaFree(dev_C);
	cudaFree(dev_A);
	cudaFree(dev_B);
}
