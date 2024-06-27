#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/quality.hpp>
#include <opencv2/core/core.hpp>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace nvonnxparser;

// Logger
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// Reading function
std::vector<char> readFile(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
    {
        return buffer;
    }
    throw std::runtime_error("Failed to read file: " + filePath);
}

// Build and save TensorRT engine
void buildEngine(const std::string& onnxFilePath, const std::string& engineFilePath)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0U); // No flags
    IParser* parser = createParser(*network, gLogger);

    // Parsing
    std::vector<char> onnxModel = readFile(onnxFilePath);
    if (!parser->parse(onnxModel.data(), onnxModel.size()))
    {
        std::cerr << "ERROR: Failed to parse the ONNX file." << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return;
    }

    // Build engine
    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 28); // 256MiB

    IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);

    std::ofstream engineFile(engineFilePath, std::ios::binary);
    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());

    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;
}

// Load engine
ICudaEngine* loadEngine(const std::string& engineFilePath, IRuntime* runtime)
{
    std::vector<char> engineData = readFile(engineFilePath);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    return engine;
}

// Preprocessing function
cv::Mat preprocessImage(const std::string& imagePath, const cv::Size& inputSize)
{
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        throw std::runtime_error("Error: Unable to load image.");
    }
    cv::resize(image, image, inputSize);
    image.convertTo(image, CV_32FC3, 1.0 / 255);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Convert HWC to NCHW
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    cv::Mat nchw(3, inputSize.height * inputSize.width, CV_32F);
    for (int i = 0; i < 3; ++i)
    {
        channels[i].reshape(1, 1).copyTo(cv::Mat(1, inputSize.height * inputSize.width, CV_32F, nchw.ptr<float>(i)));
    }

    return nchw;
}

// Postprocessing function
cv::Mat postprocessImage(const cv::Mat& output, const cv::Size& originalSize)
{
    cv::Mat chw(output.size(), CV_32FC3);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i)
    {
        channels[i] = cv::Mat(output.size().height, output.size().width, CV_32F, const_cast<float*>(output.ptr<float>() + i * output.size().height * output.size().width));
    }
    cv::merge(channels, chw);

    chw.convertTo(chw, CV_8UC3, 255.0);
    cv::cvtColor(chw, chw, cv::COLOR_RGB2BGR);
    cv::resize(chw, chw, originalSize);

    return chw;
}

// Inference function
void doInference(IExecutionContext& context, float* hInput, float* hOutput, void* dInput, void* dOutput, cudaStream_t stream, int inputSize)
{
    // Input and output tensor addresses
    context.setTensorAddress("input", dInput);
    context.setTensorAddress("output", dOutput);

    // Copy input data to GPU
    cudaMemcpyAsync(dInput, hInput, inputSize, cudaMemcpyHostToDevice, stream);

    // Inference
    context.enqueueV3(stream);

    // Copy output data to CPU
    cudaMemcpyAsync(hOutput, dOutput, inputSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
}

int main()
{
    const std::string onnxFilePath = "/path/to/onnx/model.onnx";
    const std::string engineFilePath = "/path/to/trt/engine.trt";
    const std::string imagePath = "/path/to/input/image";
    const std::string outputImagePath = "/path/to/output/image";

    // Build engine
    buildEngine(onnxFilePath, engineFilePath);

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = loadEngine(engineFilePath, runtime);
    IExecutionContext* context = engine->createExecutionContext();

    // Define input and output shapes
    const int inputChannels = 3;
    const int inputHeight = 1024;
    const int inputWidth = 1024;
    const int inputSize = inputChannels * inputHeight * inputWidth * sizeof(float);

    // Allocate host and device buffers
    float* hInput = new float[inputSize];
    float* hOutput = new float[inputSize];
    void* dInput;
    void* dOutput;
    cudaMalloc(&dInput, inputSize);
    cudaMalloc(&dOutput, inputSize);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Preprocessing
    cv::Mat inputImage = preprocessImage(imagePath, cv::Size(inputWidth, inputHeight));
    memcpy(hInput, inputImage.ptr<float>(), inputSize);

    // Inference
    doInference(*context, hInput, hOutput, dInput, dOutput, stream, inputSize);

    // Load clean image to get its shape
    cv::Mat originalImage = cv::imread(imagePath);
    cv::Size originalSize = originalImage.size();

    // Postprocessing
    cv::Mat outputImage = cv::Mat(inputHeight, inputWidth, CV_32FC3, hOutput);
    cv::Mat resultImage = postprocessImage(outputImage, originalSize);

    cv::imwrite(outputImagePath, resultImage);

    std::cout << "Output image saved to " << outputImagePath << std::endl;

    cv::Scalar mse = cv::quality::QualityMSE::compute(originalImage, resultImage, cv::noArray());
    cv::Scalar psnr = cv::quality::QualityPSNR::compute(originalImage, resultImage, cv::noArray());
    cv::Scalar ssim = cv::quality::QualitySSIM::compute(originalImage, resultImage, cv::noArray());

    std::cout << "\nMSE: " << mse[0] << std::endl;
    std::cout << "PSNR: " << psnr[0] << std::endl;
    std::cout << "SSIM: " << ssim[0] << std::endl;

    delete[] hInput;
    delete[] hOutput;
    cudaFree(dInput);
    cudaFree(dOutput);
    cudaStreamDestroy(stream);

    delete context;
    delete engine;
    delete runtime;

    return 0;
}
