#pragma once

//#ifndef FEEDBACK_EXPANSION_H
//#define FEEDBACK_EXPANSION_H

#include "Common.h"
#include "FeedbackExpansionK.cuh"

void runExactExpansion(configuration& config);

MapElement* computeCoverage(const bool* boundary, const std::vector<int>& sourceDistribution, configuration& config, int rows, int cols);

//#endif

