# Image-Dehazing
Project on solving the problem of image dehazing using neural networks

## Terms of reference
There is a classical diapositive algorithm based on the calculation of the transmittance map t(x) of the image. This map takes a value between 0 and 1, where 0 is light completely not transmitted, 1 is light completely transmitted. This means that the pixels that are strongly affected by haze have the lowest t(x) value. Next, a clean image is created based on the current pixel value I(x), the transmittance map value at that pixel t(x), and the global illumination A.

From left to right, the obscured image I(x), the clean image J(x), and the transmission map of the obscured image t(x):

![example_4_orig_algorithm.png](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_4_orig_algorithm.png)

The formula for calculating the clean image J(x) is as follows: 

`J(x) = A - (A - I(x)) / t(x)`.

As can be seen from the formula, calculation of a clean image is a rather time-consuming process. Calculating the transmittance map of each pixel and then assembling a clean image takes a lot of time, especially if we deal with high-resolution images. Therefore, the task of approximating this algorithm with the help of a neural network has arisen.

## Neural Network Architecture
AOD-Net (All-in-One Dehazing Network) was adopted as the neural network architecture. Original article: https://sites.google.com/site/boyilics/website-builder/project-page. Architecture: 

![AOD_architecture.png](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/AOD_architecture.png)

This model generates a map K(x) that replaces t(x) and A. By using convolution layers, the speed of image processing is greatly improved.

The images from the original article, [hazy images](https://drive.google.com/file/d/17ZWJOpH1AsYQhoqpWR6PK61HrUhArdAK/view) and [clear images](https://drive.google.com/file/d/1Sz5ZFFZXo3sY85R3v7yJa6W6riDGur46/view), were used as the dataset.

## Train Configuration
The model is written using the PyTorch deep learning framework in Python. The basic training config is as follows:

- `num_epochs` = 10 (number of training epochs)
- `lr` = 0.0001 (learning rate)
- `train_batch_size` = 8 (training batch size)
- `val_batch_size` = 8 (validation batch size)

MSE (mean squared error) was used as a loss function. In this task there are no requirements on quality metrics, so the result of the model is evaluated visually - whether the haze is removed well or not.

## Learing results
As a result, in spite of a small number of epochs, the result is excellent. The model separates clean image from haze well and runs much faster than the original algorithm. Examples from the test sample:

![example_1.jpg](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_1.jpg)
![example_2.jpg](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_2.jpg)
![example_3.jpg](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_3.jpg)

## Инференс модели
The model inference was performed in C++ 17 using the TensorRT framework version 10.1. The build also includes OpenCV 4.10, CUDA 12.3 and cuDNN 9.2. For inference, the model has been ported to ONNX format. Using such tools allows the model to run faster than running it in PyTorch, as well as more detailed memory handling if the model is embedded in a loaded service. Comparison of PyTorch and TensorRT throughput:

![tensorrt_vs_pytorch.png](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/tensorrt_vs_pytorch.png)

## How to run

### Python

1. Open the `test_model.ipynb` notebook. 
2. Download the necessary libraries. 
3. Specify to the `dehaze` function the path to the .pt model and the path to save images (original image and processed image). When calling the function, specify the path to the required image. 
4. If necessary, display `plt.imshow` or save `plt.imwrite` the processed snapshot.

### Python TensorRT API

1. Open `inference.py` from the `python inference` folder. 
2. Make sure you have compatible versions of TensorRT, OpenCV, and PyCuda installed. 
3. In `onnx_file_path` specify the path to the .onnx model. Create an engine file in .trt format and specify the path to it in `engine_file_path`. 
4. In `image_path` and `output_image_path`, specify the path to the input snap and specify the path to the output snap.

### C++ TensorRT API

1. Open `CMakeLists.txt` from the `cpp inference` folder. 
2. Make sure you have compatible versions of TensorRT, OpenCV, CUDA, and cuDNN installed. 
3. Specify the necessary paths and add directives. 
4. Open `main.cpp`. In `onnxFilePath` specify the file to ONNX model. 
5. Create an engine file in .trt format and specify the path to it in `engineFilePath`. 
6. In `imagePath` and `outputImagePath`, specify the path to the input snapshot and specify the path to the output snapshot. 
7. The ONNX model is configured for input and output tensor of dimensionality `1 * 3 * 1024 * 1024`, so in preprocessing the input image is compressed to a resolution of `1024 * 1024`, and in postprocessing it is returned to the original.





