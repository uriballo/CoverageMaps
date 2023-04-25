#pragma once

#ifndef FEEDBACK_EXPANSION_H
#define FEEDBACK_EXPANSION_H

#include "Common.h"
#include "FeedbackExpansionK.cuh"

void runExactExpansion(config configuration);

CUDAPair<float, int>* computeCoverage(const bool* boundary, const int* sourceDistribution, float radius, int rows, int cols, int numSources);

#endif

