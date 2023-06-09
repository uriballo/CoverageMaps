#include "Common.h"

cv::Mat IO::readImage(const std::string path, cv::ImreadModes mode) {
	// Load image in grayscale mode
	cv::Mat img = cv::imread(path, mode);

	if (img.empty()) {
		// If the image is not loaded correctly, throw an exception
		throw std::runtime_error("Could not read image: " + path);
	}

	return img;
}

float* IO::convertCV2Float(const cv::Mat& image) {
	// Check if the matrix is grayscale and of type uchar
	if (image.channels() != 1 || image.type() != CV_8U) {
		std::cerr << "Error: matrix is not grayscale or not of type uchar" << std::endl;
		return nullptr;
	}

	// Create a new float array to store the converted values
	float* floatArray = new float[image.rows * image.cols];

	// Convert the matrix to a float array
	int index = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			floatArray[index++] = static_cast<float>(image.at<uchar>(i, j));
		}
	}

	return floatArray;
}

void IO::writeBoolMatrix(bool* mat, int rows, int cols, std::string fileName, std::string path, std::string extension)
{
	std::ofstream outFile;
	outFile.open(path + fileName + extension);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outFile << std::setw(1) << mat[i * cols + j] << " ";
		}
		outFile << std::endl;
	}
	outFile.close();
}

void IO::writeIntMatrix(int* mat, int rows, int cols, std::string fileName, std::string path, std::string extension)
{
	std::ofstream outFile;
	outFile.open(path + fileName + extension);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outFile << std::setw(2) << mat[i * cols + j] << " ";
		}
		outFile << std::endl;
	}
	outFile.close();
}

void IO::writeFloatMatrix(float* mat, int rows, int cols, std::string fileName, std::string path, std::string extension) {
	std::ofstream outFile;
	outFile.open(path + fileName + extension);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outFile << std::setprecision(4) << std::setw(5) << mat[i * cols + j] << " ";
		}
		outFile << std::endl;
	}
	outFile.close();
}

void IO::writeCoverageMap(const MapElement* distanceMap, int rows, int cols, std::string fileName, std::string path, std::string extension) {
	std::ofstream outFileD;
	std::ofstream outFileS;
	outFileD.open(path + fileName + "_Distances" + extension);
	outFileS.open(path + fileName + "_Predecessors" + extension);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outFileD << std::setprecision(4) << std::setw(5) << distanceMap[i * cols + j].distance << " ";
			if (distanceMap[i * cols +j].predecessor != -1  && distanceMap[distanceMap[i * cols + j].predecessor].distance == 0)
				outFileS << std::setprecision(4) << std::setw(8) << distanceMap[i * cols + j].predecessor << " ";
			else 
				outFileS << std::setprecision(4) << std::setw(8) << "-1";
		}
		outFileD << std::endl;
		outFileS << std::endl;
	}
	outFileD.close();
	outFileS.close();
}

cv::Mat IO::convertFloat2CV(const float* array, int rows, int cols) {
	// Create a new cv::Mat matrix of type CV_32F
	cv::Mat mat(rows, cols, CV_32FC1);

	memcpy(mat.data, array, rows * cols * sizeof(float));

	return mat;
}

cv::Mat IO::createRGBImage(float* outputR, float* outputG, float* outputB, int numRows, int numCols) {
	// Initialize the output matrices.
	cv::Mat outputImage(numRows, numCols, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));

	// Copy the data from the output arrays to the corresponding channels of the output matrix.
	std::vector<cv::Mat> channels;
	channels.push_back(cv::Mat(numRows, numCols, CV_32FC1, outputR));
	channels.push_back(cv::Mat(numRows, numCols, CV_32FC1, outputG));
	channels.push_back(cv::Mat(numRows, numCols, CV_32FC1, outputB));
	cv::merge(channels, outputImage);

	return outputImage;
}

void IO::writeRGBImage(const cv::Mat& imageRGB, std::string filePath) {
	// Convert the floating-point image to 8-bit integer
	cv::Mat image8Bit;
	cv::cvtColor(imageRGB, image8Bit, cv::COLOR_RGB2BGR);
	image8Bit.convertTo(image8Bit, CV_8U, 255.0);

	// Save the image as a PNG file
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9); // Maximum compression
	cv::imwrite(filePath, image8Bit, compression_params);
}

std::vector<int> UTILS::convertString2IntVector(const std::string& str) {
	std::vector<int> result;
	std::istringstream iss(str);
	std::string token;

	while (std::getline(iss, token, ',')) {
		result.push_back(std::stoi(token));
	}

	return result;
}

std::string UTILS::convertIntVector2String(const std::vector<int>& vec) {
	std::string result = "{ ";

	for (size_t i = 0; i < vec.size(); ++i) {
		if (i % 2 == 0) {
			result += "(" + std::to_string(vec[i]) + ",";
		}
		else {
			result += std::to_string(vec[i]) + ")";
			if (i != vec.size() - 1) {
				result += ", ";
			}
		}
	}
	result += " }";
	return result;

}

std::vector<int> UTILS::generateRandomDistribution(const int* boundary, int rows, int cols, int N) {
	std::vector<int> sources;
	sources.reserve(2 * N); // Reserve space for efficiency

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> colDistribution(0, cols - 1);
	std::uniform_int_distribution<> rowDistribution(0, rows - 1);

	int i = 0;

	while (i < 2 * N) {
		int x = colDistribution(gen);
		int y = rowDistribution(gen);
		int index = y * cols + x;

		if (boundary[index] > -1) {
			sources.push_back(x);
			sources.push_back(y);
			i += 2;
		}
	}

	return sources;
}

cv::Mat UTILS::processResultsRGB(const int* boundary, const MapElement* coverageMap, const int* sourceDistribution, int numSources, const float radius, int rows, int cols) {
	int numElements = rows * cols;

	int* deviceBoundary;
	CUDA::allocateAndCopy(deviceBoundary, boundary, numElements);

	MapElement* deviceCoverageMap;
	CUDA::allocateAndCopy(deviceCoverageMap, coverageMap, numElements);

	int* deviceSourceDistribution;
	CUDA::allocateAndCopy(deviceSourceDistribution, sourceDistribution, numSources * 2);

	float* outputR;
	float* outputG;
	float* outputB;
	CUDA::allocate(outputR, numElements);
	CUDA::allocate(outputG, numElements);
	CUDA::allocate(outputB, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	processResultsRGB_ << < blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceCoverageMap, deviceSourceDistribution, numSources, outputR, outputG, outputB, radius, numElements, cols);
	CUDA::sync();

	float* R = new float[numElements];
	float* G = new float[numElements];
	float* B = new float[numElements];

	CUDA::copyDeviceToHost(R, outputR, numElements);
	CUDA::copyDeviceToHost(G, outputG, numElements);
	CUDA::copyDeviceToHost(B, outputB, numElements);

	cv::Mat colorMap = IO::createRGBImage(R, G, B, rows, cols);

	CUDA::free(outputR);
	CUDA::free(outputG);
	CUDA::free(outputB);
	CUDA::free(deviceBoundary);
	CUDA::free(deviceCoverageMap);
	CUDA::free(deviceSourceDistribution);

	return colorMap;
}

void UTILS::freeExpansionResources(cudaTextureObject_t domainTexture, cudaArray* domainArray, MapElement* deviceCoverageMap, int* domain) {
	cudaDestroyTextureObject(domainTexture);
	cudaFreeArray(domainArray);
	CUDA::free(deviceCoverageMap);
	delete[] domain;
}


