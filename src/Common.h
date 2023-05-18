#pragma once

//#ifndef COMMON_H
//#define COMMON_H

#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <random>
#include <filesystem>
#include "MapElement.cuh"

#include "CommonCUDA.h"

struct configuration {
	std::string imagePath;
	std::string imageName;

	bool storeBoundary;
	bool storeIterCoverage;

	int numberOfServices;
	float serviceRadius;

	bool customDistribution;
	std::string serviceDistribution;

	bool euclideanExpansion;
	bool exactExpansion;

	bool maxCoverage;

	std::string solutionData;
};

namespace IO {
	bool* extractImageBoundary(const std::string filePath, int& rows, int& cols);

	int* preProcessDomainImage(const std::string filePath, int& rows, int& cols);

	cv::Mat readImage(const std::string path, cv::ImreadModes mode);

	void writeBoolMatrix(bool* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	void writeIntMatrix(int* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	void writeFloatMatrix(float* mat, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	void writeCoverageMap(const MapElement* distanceMap, int rows, int cols, std::string fileName, std::string path = "output/", std::string extension = ".txt");

	float* convertCV2Float(const cv::Mat& image);

	cv::Mat convertFloat2CV(const float* array, int rows, int cols);

	cv::Mat createRGBImage(float* outputR, float* outputG, float* outputB, int numRows, int numCols);

	void writeRGBImage(const cv::Mat& imageRGB, std::string filePath);
}

namespace UTILS {
	std::vector<int> convertString2IntVector(const std::string& str);

	std::vector<int> generateRandomDistribution(const int* boundary, int rows, int cols, int N);

	cv::Mat processResultsRGB(const int* boundary, const MapElement* coverageMap, const int* sourceDistribution, int numSources, const float radius, int rows, int cols);
}

//#endif


