#include "MCLPSolver.h" 

MCLPSolver::MCLPSolver(int* domain, MapElement* coverageMap, int numServices, int rows, int cols, float radius, OptimizationParameters settings)
	: _domain(domain), _coverageMap(coverageMap), _numServices(numServices), _populationSize(settings.populationSize), _rows(rows), _cols(cols),
	_radius(radius), _numGenerations(settings.numGenerations), _mutationRate(settings.mutationRate), _stopThreshold(settings.stopThreshold) {

	generateInitialPopulation();
	_domainTexture = COVERAGE::convertDomainToTexture(_domain, &_domainArray, _rows, _cols);
}

Individual MCLPSolver::solve(){
	int currentGeneration = 0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);
	
	updatePopulationFitness();
	sortPopulation();

	Individual fittestIndividual = _population[0];

	std::cout << "[>] Initiating evolution process\n" << std::endl;

	while (currentGeneration < _numGenerations && fittestIndividual.fitness < _stopThreshold) {
		std::cout << "[~] Generation " << currentGeneration + 1 << ":" << std::endl;

		int numParents = _populationSize % 2 == 0  ? 1 + _populationSize / 2 : _populationSize / 2;
		std::vector<Individual> parents = selection(numParents);
		std::vector<Individual> children;

		children.reserve(parents.size() / 2);
		int j = 1;
		for (int i = 0; i < numParents; i+=2) {
			Individual parent1 = parents[i];
			Individual parent2 = parents[i + 1];
			// Create a child using one-point crossover
			Individual child = crossover(parent1, parent2);

			// Mutate the child based on mutation probability
			if (dis(gen) < _mutationRate) {
				mutate(child);
			}
			children.push_back(child);
			j++;
		}

		// Mutate 10% random individuals from the population
		int numRandomMutations = static_cast<int>(std::round(0.1 * _populationSize)); // 10% of population size
		for (int i = 0; i < numRandomMutations; i++) {
			int randomIndex = std::uniform_int_distribution<>(0, _populationSize - 1)(gen);
			mutate(_population[randomIndex]);
		}

		j = 0;
		for (int i = _populationSize - children.size(); i < _populationSize; i++) {
			_population[i] = children[j];
			j++;
		}

		updatePopulationFitness();
		sortPopulation();
		fittestIndividual = _population[0];
		std::cout << "\t[!] Best individual's coverage: " << fittestIndividual.fitness << "%" << std::endl;
		++currentGeneration;
	}
	std::cout << "[!] Evolution process completed." << std::endl;
	std::cout << "\tFinal best individual's coverage: " << fittestIndividual.fitness << "%" << std::endl;

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

float MCLPSolver::evaluateFitness(std::vector<int> genes) {
	// Setup the coverageMap 
	COVERAGE::setupCoverageMap(_coverageMap, genes, _radius, _rows, _cols);

	COVERAGE::computeExactCoverageMap(_domainTexture, _coverageMap, _radius, _rows, _cols);
	
	return COVERAGE::getCoveragePercent(_domainTexture, _coverageMap, _rows, _cols);
}

void MCLPSolver::updatePopulationFitness(){
	for (int i = 0; i < _populationSize; i++) {
		if (_population[i].fitness == 0) 
			_population[i].fitness = evaluateFitness(_population[i].genes);
	}
}

std::vector<Individual> MCLPSolver::selection(int numIndividuals)
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


void MCLPSolver::mutate(Individual& individual) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> mutationIndexDist(0, individual.genes.size() - 1);
	std::uniform_real_distribution<> mutationValueDist(0.01, 0.15); // Varying mutation percentage between 1% and 15%

	int mutationIndex = mutationIndexDist(gen);
	int geneValue = individual.genes[mutationIndex];
	int mutationRange = static_cast<int>(std::round(geneValue * mutationValueDist(gen)));

	if (mutationIndex % 2 == 0) {
		// Mutate X value
		int newXValue = std::max(0, std::min(_cols, geneValue + mutationRange));
		individual.genes[mutationIndex] = newXValue;
	}
	else {
		// Mutate Y value
		int newYValue = std::max(0, std::min(_rows, geneValue + mutationRange));
		individual.genes[mutationIndex] = newYValue;
	}

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

