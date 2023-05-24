#include "FeedbackExpansion.h"

// DIRTY IMPLEMENTATION OF GENETIC ALGORITHM

struct Individual {
	std::vector<int> genes;
	float fitness;
};

struct GeneticAlgorithmParams {
	int* hostDomain;
	cudaTextureObject_t deviceDomain;

	int rows, cols;
	int numServices;

	float radius;

	MapElement* deviceCoverageMap;

	int populationSize;
};

void fitnessEval(Individual& candidate, GeneticAlgorithmParams& params) {
	params.deviceCoverageMap = initialCoverageMapGPU(candidate.genes, params.deviceCoverageMap, params.numServices, params.rows, params.cols, params.radius + FLT_MIN, -1);

	float fitness = computeCoveragePercent(params.deviceDomain, params.deviceCoverageMap, params.rows, params.cols, params.radius);

	candidate.fitness = fitness;
}

Individual tournamentSelection(const std::vector<Individual>& population, int tournamentSize) {
	int populationSize = population.size();
	Individual selectedParent;

	// Perform tournament selection
	for (int i = 0; i < tournamentSize; ++i) {
		int randomIndex = rand() % populationSize;
		const Individual& candidate = population[randomIndex];

		// Compare fitness values
		if (selectedParent.genes.empty() || candidate.fitness > selectedParent.fitness) {
			selectedParent = candidate;
		}
	}

	return selectedParent;
}

Individual generateIndividual(GeneticAlgorithmParams& params) {
//	std::cout << "\t\t about to generate a random distribution..." << std::endl;
	std::vector<int> genes = UTILS::generateRandomDistribution(params.hostDomain, params.rows, params.cols, params.numServices);
//	std::cout << "\t\t got random genes..." << std::endl;
	params.deviceCoverageMap = initialCoverageMapGPU(genes, params.deviceCoverageMap, params.numServices, params.rows, params.cols, params.radius + FLT_MIN, -1);
//	std::cout << "\t\t got initialized map...\n";
	float fitness = computeCoveragePercent(params.deviceDomain, params.deviceCoverageMap, params.rows, params.cols, params.radius);
//	std::cout << "\t\t got coverage map...\n";

	Individual i;
	i.genes = genes;
	i.fitness = fitness;

	return i;
}

// Perform crossover between two parents to generate offspring
std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
	// Choose a random crossover point
	int crossoverPoint = rand() % parent1.size();

	std::vector<int> offspring;
	offspring.reserve(parent1.size());

	// Copy genes from parent1 up to the crossover point
	for (int i = 0; i < crossoverPoint; ++i) {
		offspring.push_back(parent1[i]);
	}

	// Copy remaining genes from parent2
	for (int i = crossoverPoint; i < parent2.size(); ++i) {
		offspring.push_back(parent2[i]);
	}

	return offspring;
}

void mutate(std::vector<int>& genes, int xMax, int yMax) {
	// Choose a random gene to mutate
	int geneIndex = rand() % genes.size();

	if (geneIndex % 2 == 0)
		genes[geneIndex] += (rand() % (xMax + 1));
	else
		genes[geneIndex] += (rand() % (yMax + 1));
}

std::vector<Individual> generatePopulation(GeneticAlgorithmParams& params) {
	std::vector<Individual> population;

	population.reserve(params.populationSize);
	std::cout << "Ready to generate population..." << std::endl;
	for (int i = 0; i < params.populationSize; i++) {
		population.push_back(generateIndividual(params));
		std::cout << "\t " << i + 1 << " out of " << params.populationSize << " generated...\n";
		std::cout << "\t\t fitness: " << population[i].fitness << std::endl;
	}

	return population;
}

bool compareIndividuals(const Individual& individual1, const Individual& individual2) {
	return individual1.fitness > individual2.fitness;
}

std::vector<int> geneticAlgorithm(GeneticAlgorithmParams& params) {
	srand(time(nullptr));

	std::vector<Individual> population = generatePopulation(params);

	int populationSize = params.populationSize;
	std::cout << "Population Generated...\n";

	for (int generation = 0; generation < 10; generation++) {
		std::vector<Individual> newPopulation;
		newPopulation.reserve(populationSize);
		std::cout << "Generation " << generation << std::endl;

		// Generate new population through selection, crossover, and mutation
		for (int i = 0; i < populationSize; ++i) {
			// Select parents using tournament selection
			Individual parent1 = tournamentSelection(population, populationSize);
			Individual parent2 = tournamentSelection(population, populationSize);

			// Perform crossover to generate offspring
			std::vector<int> offspringGenes = crossover(parent1.genes, parent2.genes);

			// Perform mutation on the offspring
			if (rand() < 0.1)
				mutate(offspringGenes, params.cols, params.rows);

			// Create a new individual with the offspring genes
			Individual offspring;
			offspring.genes = offspringGenes;

			fitnessEval(offspring, params);

			newPopulation.push_back(offspring);
		}

		// Replace the old population with the new population
		population = newPopulation;
	}
	// Find the best individual in the final population
	std::sort(population.begin(), population.end(), compareIndividuals);

	for (int i = 0; i < params.populationSize; i++) {
		std::cout << "Fitness " << i << ": " << population[i].fitness << std::endl;
	}
	return population[0].genes;
}

std::vector<int> runGeneticAlgorithm(configuration& config) {
	std::cout << "Starting Genetic Algorithm...\n";
	int rows, cols;
	int* domain = IO::preProcessDomainImage(config.imagePath, rows, cols);
	int numElements = rows * cols;

	cudaArray* domainArray;
	cudaTextureObject_t texDomainObj = getDomainGPU(domain, rows, cols, &domainArray);

	GeneticAlgorithmParams params;
	std::cout << "Got domain texture...\n";

	std::vector<int> initialDist;
	initialDist.push_back(0);
	initialDist.push_back(0);

	params.rows = rows;
	params.cols = cols;
	params.deviceDomain = texDomainObj;
	params.numServices = config.numberOfServices;
	params.hostDomain = domain;
	params.deviceCoverageMap = initialCoverageMapGPU(initialDist, 1, rows, cols, 0, -1);
	params.populationSize = 10;

	std::cout << "Starting Evolution...\n";

	auto startTime = std::chrono::steady_clock::now();
	std::vector<int> bestCandidate = geneticAlgorithm(params);
	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;
	config.solutionData += "Genetic Algorithm compute time: " + std::to_string(seconds) + " (s)\n";

	cudaDestroyTextureObject(texDomainObj);
	cudaFreeArray(domainArray);

	CUDA::free(params.deviceCoverageMap);

//	delete[] params.hostDomain;
	delete[] domain;

	return bestCandidate;
}

cudaTextureObject_t getDomainGPU(const int* hostDomain, int rows, int cols, cudaArray** domainArray) {
	// Create a CUDA array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();

	CUDA::allocateArray(*domainArray, channelDesc, cols, rows);
//	cudaMallocArray(domainArray, &channelDesc, cols, rows);
//	cudaMemcpyToArray(*domainArray, 0, 0, hostDomain, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
	CUDA::copyToArray(*domainArray, hostDomain, rows * cols);

	// Create a texture object
	cudaTextureObject_t texDomainObj = 0;
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = *domainArray;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaCreateTextureObject(&texDomainObj, &resDesc, &texDesc, NULL);

	return texDomainObj;
}

float coveragePercent(cudaTextureObject_t domainTexture, std::vector<int> genes, int numServices, float radius, MapElement* deviceCoverageMap, int rows, int cols) {
	deviceCoverageMap = initialCoverageMapGPU(genes, deviceCoverageMap, numServices, rows, cols, radius + FLT_MIN, -1);

	int* deviceInteriorPoints;
	CUDA::allocateAndSet(deviceInteriorPoints, 1, 0);

	int* deviceCoveredPoints;
	CUDA::allocateAndSet(deviceCoveredPoints, 1, 0);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	evalCoverage << <blocksPerGrid, threadsPerBlock >> > (domainTexture, deviceCoverageMap, rows, cols, deviceInteriorPoints, deviceCoveredPoints);
	CUDA::sync();

	int interiorPoints, coveredPoints;
	CUDA::copyDeviceToHost(&interiorPoints, deviceInteriorPoints, 1);
	CUDA::copyDeviceToHost(&coveredPoints, deviceCoveredPoints, 1);

//	std::cout << "\t\t\t" << (static_cast<float>(coveredPoints)) << " " << (static_cast<float>(interiorPoints)) << std::endl;

	return 100* (static_cast<float>(coveredPoints) / static_cast<float>(interiorPoints));
}

void runExactExpansion(configuration& config) {
	std::vector<int> servicesDistribution = runGeneticAlgorithm(config);

	int rows, cols;
	int* domain = IO::preProcessDomainImage(config.imagePath, rows, cols);
	int numElements = rows * cols;

	cudaArray* domainArray;
	cudaTextureObject_t texDomainObj = getDomainGPU(domain, rows, cols, &domainArray);

	MapElement* deviceCoverageMap = initialCoverageMapGPU(servicesDistribution, config.numberOfServices, rows, cols, config.serviceRadius + FLT_MIN, -1);

	auto startTime = std::chrono::steady_clock::now();

	MapElement* coverageMap = computeCoverage(texDomainObj, deviceCoverageMap, config, rows, cols);// computeCoverage(domain, servicesDistribution, config, rows, cols);

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	float coverage = coveragePercent(texDomainObj, deviceCoverageMap, rows, cols);
	config.solutionData += "Coverage map compute time: " + std::to_string(seconds) + " (s)\n";
	config.solutionData += "Coverage: " + std::to_string(coverage) + "% \n";


	cv::Mat processedResultsRGB = UTILS::processResultsRGB(domain, coverageMap, servicesDistribution.data(), config.numberOfServices, config.serviceRadius, rows, cols);
	IO::writeRGBImage(processedResultsRGB, "output/" + config.imageName);

	cudaDestroyTextureObject(texDomainObj);
	cudaFreeArray(domainArray);

	CUDA::free(deviceCoverageMap);

	delete[] coverageMap;
	delete[] domain;

	/*
	int rows, cols;
	int* domain = IO::preProcessDomainImage(config.imagePath, rows, cols);
	int numElements = rows * cols;

	config.solutionData = "";
	config.solutionData += "Domain dimensions: " + std::to_string(rows) + " x " + std::to_string(cols) + " (" + std::to_string(numElements) + " pixels) \n\n";
	
	if (config.storeBoundary) {
		IO::writeIntMatrix(domain, rows, cols, "domain");
	}

	std::vector<int> servicesDistribution;

	if (config.customDistribution) {
		servicesDistribution = UTILS::convertString2IntVector(config.serviceDistribution);
	}
	else {
		servicesDistribution = UTILS::generateRandomDistribution(domain, rows, cols, config.numberOfServices);
	}

	cudaArray* domainArray;
	cudaTextureObject_t texDomainObj = getDomainGPU(domain, rows, cols, &domainArray);

	MapElement* deviceCoverageMap = initialCoverageMapGPU(servicesDistribution, config.numberOfServices, rows, cols, config.serviceRadius + FLT_MIN, -1);

	auto startTime = std::chrono::steady_clock::now();

	MapElement* coverageMap = computeCoverage(texDomainObj, deviceCoverageMap, config, rows, cols);// computeCoverage(domain, servicesDistribution, config, rows, cols);

	auto endTime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	double seconds = duration.count() / 1000.0;

	float coverage = coveragePercent(texDomainObj, deviceCoverageMap, rows, cols);
	config.solutionData += "Coverage map compute time: " + std::to_string(seconds) + " (s)\n";
	config.solutionData += "Coverage: " + std::to_string(coverage) + "% \n";


	cv::Mat processedResultsRGB = UTILS::processResultsRGB(domain, coverageMap, servicesDistribution.data(), config.numberOfServices, config.serviceRadius, rows, cols);
	IO::writeRGBImage(processedResultsRGB, "output/" + config.imageName);

	cudaDestroyTextureObject(texDomainObj);
	cudaFreeArray(domainArray);

	CUDA::free(deviceCoverageMap);

	delete[] coverageMap;
	delete[] domain;
	*/
}

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, int numServices, int rows, int cols, float initRadius, int initPredecessor) {
	int numElements = rows * cols;

	int* servicesArray = servicesDistribution.data();

	int* deviceServices;
	CUDA::allocateAndCopy(deviceServices, servicesArray, numServices * 2);

	MapElement* deviceCoverageMap;
	CUDA::allocate(deviceCoverageMap, numElements);

	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	initCoverageMap << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, initRadius, initPredecessor, deviceServices, numServices, numElements, cols);
	CUDA::sync();

	return deviceCoverageMap;
}

MapElement* initialCoverageMapGPU(std::vector<int> servicesDistribution, MapElement* deviceCoverageMap, int numServices, int rows, int cols, float initRadius, int initPredecessor) {
	int numElements = rows * cols;

	int* servicesArray = servicesDistribution.data();

	int* deviceServices;
	CUDA::allocateAndCopy(deviceServices, servicesArray, numServices * 2);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	initCoverageMap << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, initRadius, initPredecessor, deviceServices, numServices, numElements, cols);
	CUDA::sync();

	return deviceCoverageMap;
}

int* getDomainGPU(const int* hostDomain, int numElements) {
	int* deviceDomain;
	CUDA::allocateAndCopy(deviceDomain, hostDomain, numElements);

	return deviceDomain;
}

MapElement* computeCoverage(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, configuration& config, int rows, int cols) {
	bool hostFlag = false;
	bool* deviceFlag;

	int numElements = rows * cols;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	config.solutionData += "Number of threads per block: " + std::to_string(threadsPerBlock.x * threadsPerBlock.y) + "\n";
	config.solutionData += "Number of blocks: " + std::to_string(blocksPerGrid.x * blocksPerGrid.y) + "\n";

	int iterations = 1;

	CUDA::allocate(deviceFlag, 1);
	
	MapElement* intermediateResult = new MapElement[numElements];
	bool storeData = config.storeIterCoverage;

	do {
		int innerIterations = 1;

		do {
			CUDA::set(deviceFlag, false);
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (domainTexture, deviceCoverageMap, deviceFlag, rows, cols, config.serviceRadius);
			CUDA::sync();
			CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

			if (storeData) {
				CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);
				std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
				IO::writeCoverageMap(intermediateResult, rows, cols, fileName);
				innerIterations++;
			}
		} while (hostFlag);

		CUDA::set(deviceFlag, false);
		EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, deviceFlag, rows, cols, config.serviceRadius);
		CUDA::sync();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);

		if (storeData) {
			CUDA::copyDeviceToHost(intermediateResult, deviceCoverageMap, numElements);

			std::string fileName = "iteration_" + std::to_string(iterations) + "." + std::to_string(innerIterations);
			IO::writeCoverageMap(intermediateResult, rows, cols, fileName);
		}
		iterations++;

	} while (hostFlag );

	config.solutionData += "Total iterations: " + std::to_string(iterations) + "\n\n";
	
	MapElement* hostCoverageMap = new MapElement[numElements];
	CUDA::copyDeviceToHost(hostCoverageMap, deviceCoverageMap, numElements);

	//CUDA::free(deviceCoverageMap);
	CUDA::free(deviceFlag);

	return hostCoverageMap;
}

float computeCoveragePercent(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius) {
	bool hostFlag = false;
	bool* deviceFlag;

	int numElements = rows * cols;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	int iterations = 1;
	CUDA::allocate(deviceFlag, 1);
	MapElement* intermediateResult = new MapElement[numElements];

	do {
		int innerIterations = 1;

		do {
			CUDA::set(deviceFlag, false);
			euclideanExpansion << <blocksPerGrid, threadsPerBlock >> > (domainTexture, deviceCoverageMap, deviceFlag, rows, cols, radius);
			CUDA::sync();
			CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
		} while (hostFlag);

		CUDA::set(deviceFlag, false);
		EEDT << <blocksPerGrid, threadsPerBlock >> > (deviceCoverageMap, deviceFlag, rows, cols, radius);
		CUDA::sync();
		CUDA::copyDeviceToHost(&hostFlag, deviceFlag, 1);
	} while (hostFlag);

	CUDA::free(deviceFlag);

	return coveragePercent(domainTexture, deviceCoverageMap, rows, cols);
}



