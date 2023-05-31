#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>
#include "Coverage.h"
#include "FeedbackExpansionK.cuh"

struct Individual {
	std::vector<int> genes;
	float fitness;

	Individual() : fitness(0.0) {}

	Individual(std::vector<int> g) : fitness(0.0), genes(g) {};
};

class MCLPSolver {
public:
	// Assumes that coverageMap is a pointer towards GPU memory.
	MCLPSolver(int* domain, MapElement* coverageMap, int numServices, int rows, int cols, float radius, OptimizationParameters settings);

	Individual solve();
private:
	void generateInitialPopulation();

	float evaluateFitness(std::vector<int> genes);

	void updatePopulationFitness();
	
	std::vector<Individual> selection(int numIndividuals);

	std::vector<Individual> tournamentBasedSelection(int numIndividuals);

	Individual crossover(const Individual& parent1, const Individual& parent2);

	void mutate(Individual& individual);

	void sortPopulation();
	
	void freeResources();

	int* _domain;
	cudaArray* _domainArray;
	cudaTextureObject_t _domainTexture;
	MapElement* _coverageMap;
	std::vector<Individual> _population;
	
	int _numServices;
	int _rows;
	int _cols;
	float _radius;

	int _numGenerations;
	int _populationSize;
	float _mutationRate;

	float _stopThreshold;
};

