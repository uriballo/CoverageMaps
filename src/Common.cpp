#include "Common.h"

bool* IO::extractImageBoundary(const std::string filePath, int& rows, int& cols) {
	// Read the image file into a matrix.
	cv::Mat image = readImage(filePath, cv::IMREAD_GRAYSCALE);

	// Store some properties of the image in variables.
	rows = image.rows;
	cols = image.cols;
	const int numElements = rows * cols;

	// Convert the matrix to a flat array of floats.
	float* hostDomain = cvBW2FloatArray(image);

	// Allocate memory on the GPU for the raw and normalized domains.
	bool* deviceBoundary;
	CUDA::allocate(deviceBoundary, numElements);

	float* deviceDomain;
	CUDA::allocateAndCopy(deviceDomain, hostDomain, numElements);

	// Set up the dimensions for the GPU kernel.
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Call the GPU kernel to normalize the domain.
	extractBoundary << <blocksPerGrid, threadsPerBlock >> > (deviceDomain, deviceBoundary, numElements);

	// Wait for the GPU kernel to finish.
	CUDA::synchronize();

	bool* result = new bool[numElements];
	//	GPU::copyDeviceToHost(result, normalizedDomain, pixels);
	CUDA::copyDeviceToHost(result, deviceBoundary, numElements);

	// Free memory on the GPU.
	CUDA::free(deviceDomain);
	CUDA::free(deviceBoundary);

	// Return the result array.
	return result;
}

cv::Mat IO::readImage(const std::string path, cv::ImreadModes mode) {
	// Load image in grayscale mode
	cv::Mat img = cv::imread(path, mode);

	if (img.empty()) {
		// If the image is not loaded correctly, throw an exception
		throw std::runtime_error("Could not read image: " + path);
	}

	return img;
}

float* IO::cvBW2FloatArray(const cv::Mat& image) {
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

cv::Mat IO::floatToCV(const float* array, int rows, int cols) {
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
	channels.push_back(cv::Mat(numRows, numCols, CV_32FC1, outputB));
	channels.push_back(cv::Mat(numRows, numCols, CV_32FC1, outputG));
	channels.push_back(cv::Mat(numRows, numCols, CV_32FC1, outputR));
	cv::merge(channels, outputImage);

	return outputImage;
}


void IO::storeBWImage(const cv::Mat& mat, std::string fileName) {
	cv::Mat floatMat8UC1;
	mat.convertTo(floatMat8UC1, CV_8UC1, 255.0);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9); // PNG compression level, 0-9 (9 being highest compression)

	std::string filePath = "output/" + fileName + ".png";
	cv::imwrite(filePath, floatMat8UC1, compression_params);
}

void IO::showRGB(const cv::Mat& outputImage){
	cv::imwrite("output/ie.png", outputImage);
	cv::waitKey(0);
}

void IO::showHeatMap(const float* heatMap, int rows, int cols) {
	cv::Mat aux = floatToCV(heatMap, rows, cols);

	cv::Mat normalized;
	cv::normalize(aux, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	cv::Mat heatmapN;
	cv::applyColorMap(normalized, heatmapN, cv::COLORMAP_HOT);

	cv::imshow("Heat Map", heatmapN);
	cv::waitKey(0);
}

int* UTILS::getRandomSourceDistribution(bool* boundary, int rows, int cols, int N) {
	int* sources = new int[2 * N];

	std::random_device rd; // obtain a random seed from the hardware
	std::mt19937 gen(rd()); // seed the generator
	std::uniform_int_distribution<> colDistribution(0, cols - 1); // define the range for the column index
	std::uniform_int_distribution<> rowDistribution(0, rows - 1); // define the range for the row index

	int i = 0;

	while (i < 2 * N) {
		int x = colDistribution(gen);
		int y = rowDistribution(gen);
		int index = y * cols + x;

		if (!boundary[index]) {
			sources[i] = x;
			sources[i + 1] = y;
			i += 2;
		}
	}

	return sources;
}

float* UTILS::processResults(const bool* boundary, const float* coverageMap, const float radius, int rows, int cols) {
	const int numElements = rows * cols;

	float* deviceOutput;
	CUDA::allocate(deviceOutput, numElements);

	float* deviceCoverage;
	CUDA::allocateAndCopy(deviceCoverage, coverageMap, numElements);

	bool* deviceDomain;
	CUDA::allocateAndCopy(deviceDomain, boundary, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	processResultsBW << < blocksPerGrid, threadsPerBlock >> > (deviceDomain, deviceCoverage, deviceOutput, radius, rows * cols);
	CUDA::synchronize();

	float* result = new float[numElements];

	CUDA::copyDeviceToHost(result, deviceOutput, numElements);

	CUDA::free(deviceOutput);
	CUDA::free(deviceCoverage);
	CUDA::free(deviceDomain);

	return result;
}

float* UTILS::processResults_(const bool* boundary, const CUDAPair<float, int>* coverageMap, const float radius, int rows, int cols) {
	const int numElements = rows * cols;

	float* deviceOutput;
	CUDA::allocate(deviceOutput, numElements);

	CUDAPair<float, int>* deviceCoverage;
	CUDA::allocateAndCopy(deviceCoverage, coverageMap, numElements);

	bool* deviceDomain;
	CUDA::allocateAndCopy(deviceDomain, boundary, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	processResultsBW_ << < blocksPerGrid, threadsPerBlock >> > (deviceDomain, deviceCoverage, deviceOutput, radius, rows * cols);
	CUDA::synchronize();

	float* result = new float[numElements];

	CUDA::copyDeviceToHost(result, deviceOutput, numElements);

	CUDA::free(deviceOutput);
	CUDA::free(deviceCoverage);
	CUDA::free(deviceDomain);

	return result;
}

cv::Mat UTILS::processResultsRGB(const bool* boundary, const int* sources, const CUDAPair<float, int>* coverageMap, const float radius, int rows, int cols, int numSources) {
	int numElements = rows * cols;
	
	bool* deviceBoundary;
	CUDA::allocateAndCopy(deviceBoundary, boundary, numElements);
	
	int* deviceSources;
	CUDA::allocateAndCopy(deviceSources, sources, numSources);

	CUDAPair<float, int>* deviceCoverageMap;
	CUDA::allocateAndCopy(deviceCoverageMap, coverageMap, numElements);

	float* outputR;
	float* outputG; 
	float* outputB;
	CUDA::allocate(outputR, numElements);
	CUDA::allocate(outputG, numElements);
	CUDA::allocate(outputB, numElements);

	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

	processResultsRGB_ << < blocksPerGrid, threadsPerBlock >> > (deviceBoundary, deviceSources, deviceCoverageMap, outputR, outputG, outputB, radius, numElements, cols);
	CUDA::synchronize();

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

	return colorMap;
}

void UTILS::initializeCoverageMap(CUDAPair<float, int>* coverageMap, const float initDist, const int initPredecessor, const int size) {
	for (int i = 0; i < size; i++)
		coverageMap[i] = CUDAPair<float, int>{ initDist, initPredecessor };
}

void UTILS::initializeSources(CUDAPair<float, int>* coverageMap, const int* sourceDistribution, const int numSources, const int cols)
{
	int j = -1;
	for (int i = 0; i < 2 * numSources; i += 2) {
		int index = sourceDistribution[i + 1] * cols + sourceDistribution[i];
		coverageMap[index].first = 0.0f;
		coverageMap[index].second = index;
	}
}
