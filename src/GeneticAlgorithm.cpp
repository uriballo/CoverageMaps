#include "GeneticAlgorithm.h" 

MCLPSolver::MCLPSolver(int* domain, MapElement* coverageMap, int numServices, int rows, int cols, float radius, OptimizationParameters settings)
	: _domain(domain), _coverageMap(coverageMap), _numServices(numServices), _populationSize(settings.populationSize), _rows(rows), _cols(cols),
	_radius(radius), _numGenerations(settings.numGenerations), _mutationRate(settings.mutationRate), _stopThreshold(settings.stopThreshold) {

	generateInitialPopulation();
	initializeTexture();
}

Individual MCLPSolver::solve(){
	int currentGeneration = 0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);
	
	evaluatePopulation();
	sortPopulation();

	Individual fittestIndividual = _population[0];

	std::cout << "Initiating evolution process" << std::endl;
	while (currentGeneration < _numGenerations && fittestIndividual.fitness < _stopThreshold) {
		std::cout << "\tGeneration " << currentGeneration << std::endl;

		int numParents = _populationSize % 2 == 0  ? 1 + _populationSize / 2 : _populationSize / 2;
		std::vector<Individual> parents = selection(numParents);
	//	std::cout << "\t\t parents selected: " << parents.size() << " " <<numParents << std::endl;
		std::vector<Individual> children;

		children.reserve(parents.size() / 2);
		int j = 1;
		for (int i = 0; i < numParents; i+=2) {
			Individual parent1 = parents[i];
			Individual parent2 = parents[i + 1];
		//	std::cout << "\t\tcreating child " << j << " out of " << numParents / 2 << std::endl;
			// Create a child using one-point crossover
			Individual child = crossover(parent1, parent2);

			// Mutate the child based on mutation probability
			if (dis(gen) < _mutationRate) {
			//	std::cout << "\t\t\t mutating child genes" << std::endl;
				mutate(child);
			}
		//	child.fitness = evaluateFitness(child.genes);
		//	std::cout << "\t\t\t child fitness is " << child.fitness <<std::endl;
			children.push_back(child);
			j++;

		}

	//	std::cout << "\t\tall children created" << std::endl;

		// Less fit individuals are replaced by children
		j = 0;
		for (int i = _populationSize - children.size(); i < _populationSize; i++) {
			_population[i] = children[j];
			j++;
		}
		//std::cout << "\t\treplaced old chaps" << std::endl;
		evaluatePopulation();
		sortPopulation();
		fittestIndividual = _population[0];
		std::cout << "\t\tbest fitness: " << fittestIndividual.fitness << "%" << std::endl;
		++currentGeneration;
	}

	return fittestIndividual;
}

void MCLPSolver::generateInitialPopulation(){
	_population.reserve(_populationSize);
	std::cout << "Initializing population..." << std::endl;
	for (int i = 0; i < _populationSize; i++) {
		//std::cout << "\t generating " << i << " out of " << _populationSize << " individuals" << std::endl;
		std::vector<int> genes = UTILS::generateRandomDistribution(_domain, _rows, _cols, _numServices);

		_population.push_back(Individual(genes));
	}
}

void MCLPSolver::initializeTexture(){
	_domainTexture = UTILS::convertDomain2Texture(_domain, _rows, _cols, &_domainArray);
}

void MCLPSolver::reinitializeCoverageMap(std::vector<int> genes){
	int numElements = _rows * _cols;

	int* servicesArray = genes.data();

	int* deviceServices;
	CUDA::allocateAndCopy(deviceServices, servicesArray, _numServices * 2);


	dim3 threadsPerBlock(32, 32);
	dim3 blocksPerGrid((_cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	initCoverageMap << <blocksPerGrid, threadsPerBlock >> > (_coverageMap, _radius + FLT_MIN, -1, deviceServices, _numServices, numElements, _cols);
	CUDA::sync();
}

void simpleCoverage(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols, float radius) {
	bool hostFlag = false;
	bool* deviceFlag;

	int numElements = rows * cols;

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	CUDA::allocate(deviceFlag, 1);

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
}

float coveragePercent(cudaTextureObject_t domainTexture, MapElement* deviceCoverageMap, int rows, int cols) {

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


	CUDA::free(deviceInteriorPoints);
	CUDA::free(deviceCoveredPoints);

	return 100 * (static_cast<float>(coveredPoints) / static_cast<float>(interiorPoints));
}

float MCLPSolver::evaluateFitness(std::vector<int> genes) {
	// Setup the coverageMap 
	reinitializeCoverageMap(genes);

	simpleCoverage(_domainTexture, _coverageMap, _rows, _cols, _radius);

	return coveragePercent(_domainTexture, _coverageMap, _rows, _cols);
}

void MCLPSolver::evaluatePopulation(){
	for (int i = 0; i < _populationSize; i++) {
		if (_population[i].fitness == 0) 
			_population[i].fitness = evaluateFitness(_population[i].genes);
	}
}

std::vector<Individual> MCLPSolver::selection(int numIndividuals) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	std::vector<Individual> selectedIndividuals;
	selectedIndividuals.reserve(numIndividuals);

	// Calculate total fitness of the population
	double totalFitness = 0.0;
	for (const auto& individual : _population) {
		totalFitness += individual.fitness;
	}

	// Spin the roulette wheel and select individuals
	for (int i = 0; i < numIndividuals; ++i) {
		double spin = dis(gen) * totalFitness;
		double currentFitness = 0.0;

		for (const auto& individual : _population) {
			currentFitness += individual.fitness;
			if (currentFitness >= spin) {
				selectedIndividuals.push_back(individual);
				break;
			}
		}
	}

	return selectedIndividuals;
}

std::vector<Individual> MCLPSolver::tournamentBasedSelection(int numIndividuals)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, _populationSize - 1);

	std::vector<Individual> selectedIndividuals;
	selectedIndividuals.reserve(numIndividuals);

	for (int i = 0; i < numIndividuals; ++i) {
		// Perform tournament selection
		int tournamentSize = 3;
		std::vector<int> tournamentIndices(tournamentSize);

		// Select random individuals for the tournament
		for (int j = 0; j < tournamentSize; ++j) {
			tournamentIndices[j] = dis(gen);
		}

		// Find the individual with the highest fitness in the tournament
		Individual winner = _population[tournamentIndices[0]];
		for (int j = 1; j < tournamentSize; ++j) {
			Individual& contender = _population[tournamentIndices[j]];
			if (contender.fitness > winner.fitness) {
				winner = contender;
			}
		}

		// Add the winner to the selected individuals
		selectedIndividuals.push_back(winner);
	}

	return selectedIndividuals;
}


Individual MCLPSolver::crossover(const Individual& parent1, const Individual& parent2) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, parent1.genes.size() - 1);

	// Select a random crossover point
	int crossoverPoint = dis(gen);

	// Create a child individual
	Individual child;
	child.genes.reserve(parent1.genes.size());

	// Copy genes from parent1 up to the crossover point
	for (int i = 0; i < crossoverPoint; ++i) {
		child.genes.push_back(parent1.genes[i]);
	}

	// Copy genes from parent2 after the crossover point
	for (int i = crossoverPoint; i < parent1.genes.size(); ++i) {
		child.genes.push_back(parent2.genes[i]);
	}

	return child;
}

/*
Individual MCLPSolver::crossover(const Individual& parent1, const Individual& parent2){
	Individual child;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(0, 1);

	int numGenes = parent1.genes.size();

	for (int i = 0; i < numGenes; ++i) {
		if (dis(gen) == 0) {
			child.genes.push_back(parent1.genes[i]);
		}
		else {
			child.genes.push_back(parent2.genes[i]);
		}
	}

	return child;
}
*/


void MCLPSolver::mutate(Individual& individual){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> mutationIndexDist(0, individual.genes.size() - 1);
	std::uniform_int_distribution<> mutationXValueDist(0, _cols);
	std::uniform_int_distribution<> mutationYValueDist(0, _rows);

	int mutationIndex = mutationIndexDist(gen);

	if (mutationIndex % 2 == 0)
		individual.genes[mutationIndex] = mutationXValueDist(gen);
	else
		individual.genes[mutationIndex] = mutationYValueDist(gen);

	individual.fitness = 0.0f;
}

void MCLPSolver::sortPopulation(){
	std::sort(_population.begin(), _population.end(), [](const Individual& a, const Individual& b) {
		return a.fitness > b.fitness;
		});
}

void MCLPSolver::freeResources()
{
	UTILS::freeExpansionResources(_domainTexture, _domainArray, _coverageMap, _domain);
}

