# Image-Dehazing
Project on solving the problem of image dehazing using neural networks

## Terms of reference
There is a classical diapositive algorithm based on the calculation of the transmittance map t(x) of the image. This map takes a value between 0 and 1, where 0 is light completely not transmitted, 1 is light completely transmitted. This means that the pixels that are strongly affected by haze have the lowest t(x) value. Next, a clean image is created based on the current pixel value I(x), the transmittance map value at that pixel t(x), and the global illumination A.

Слева направо - затуманенный снимок (I(x)), чистый снимок (J(x)), карта пропускания затуманенного снимка (t(x)):

![example_4_orig_algorithm.png](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_4_orig_algorithm.png)

The formula for calculating the clean image J(x) is as follows: 

`J(x) = A - (A - I(x)) / t(x)`.

As can be seen from the formula, calculation of a clean image is a rather time-consuming process. Calculating the transmittance map of each pixel and then assembling a clean image takes a lot of time, especially if we deal with high-resolution images. Therefore, the task of approximating this algorithm with the help of a neural network has arisen.

## Neural Network Architecture
В качестве архитектуры нейронной сети была взята AOD-Net (All-in-One Dehazing Network). Оригинальная статья: https://sites.google.com/site/boyilics/website-builder/project-page. Архитектура: 

![AOD_architecture.png](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/AOD_architecture.png)

Данная модель формирует карту K(x), которая заменяет t(x) и A. За счет использования сверточных слоёв скорость обработки снимка сильно повышается.

В качестве датасета использовались снимки из оригинальной статьи - [hazy images](https://drive.google.com/file/d/17ZWJOpH1AsYQhoqpWR6PK61HrUhArdAK/view) и [clear images](https://drive.google.com/file/d/1Sz5ZFFZXo3sY85R3v7yJa6W6riDGur46/view).

## Train Configuration
Модель написана с помощью фреймворка глубокого обучения PyTorch в Python. Основной тренировочный конфиг такой:

- `num_epochs` = 10 (количество эпох обучения)
- `lr` = 0.0001 (скорость обучения)
- `train_batch_size` = 8 (размер тренировочного батча)
- `val_batch_size` = 8 (размер оценочного батча)

В качестве лосс-функции использовался MSE (mean squared error). В данной задаче нет каких-то требований по метрикам качества, поэтому результат модели оценивается зрительно - хорошо удалена дымка или нет.

## Результат обучения
В результате, несмотря на небольшое число эпох, получилось добиться отличного результата. Модель хорошо отделяет чистый снимок от дымки и работает намного быстрее оригинального алгоритма. Примеры с тестовой выборки:

![example_1.jpg](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_1.jpg)
![example_2.jpg](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_2.jpg)
![example_3.jpg](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/example_3.jpg)

## Инференс модели
Инференс модели выполнялся в C++ 17 с помощью фреймворка TensorRT версии 10.1. Также в билд входят OpenCV 4.10, CUDA 12.3 и cuDNN 9.2. Для инференса модель была перенесена в ONNX формат. Использование таких инструментов позволяет ускорить работу модели по сравнению с запуском в PyTorch, а также более детально работать с памятью, если модель встроена в нагруженный сервис. Сравнение пропускной способности PyTorch и TensorRT:

![tensorrt_vs_pytorch.png](https://github.com/Shkraboom/Image-Dehazing/blob/main/data/examples/tensorrt_vs_pytorch.png)

## Как запустить

### Python

1. Откройте блокнот `test_model.ipynb`. 
2. Загрузите необходимые библиотеки. 
3. Укажите функции `dehaze` путь к модели .pt и путь для сохранения изображений (исходный снимок и обработанный). При вызове функции укажите путь до нужного снимка. 
4. При необходимости отобразите `plt.imshow` или сохраните `plt.imwrite` обработанный снимок.

### Python TensorRT API

1. Откройте `inference.py` из папки `python inference`. 
2. Убедитесь, что у вас установлены совместимые версии TensorRT, OpenCV и PyCuda. 
3. В `onnx_file_path` укажите путь к модели .onnx. Создайте файл движка в формате .trt и укажите путь до него в `engine_file_path`. 
4. В `image_path` и `output_image_path` укажите путь до входного сника и задайте путь до выходного.

### C++ TensorRT API

1. Откройте `CMakeLists.txt` из папки `cpp inference`. 
2. Убедитесь, что у вас установлены совместимые версии TensorRT, OpenCV, CUDA и cuDNN. 
3. Укажите необходимые пути и добавьте директивы. 
4. Откройте `main.cpp`. В `onnxFilePath` укажите файл к ONNX модели. 
5. Создайте файл движка в формате .trt и укажите путь до него в `engineFilePath`. 
6. В `imagePath` и `outputImagePath` укажите путь до входного сника и задайте путь до выходного. 
7. Модель ONNX настроена на входной и выходной тензор размерности `1 * 3 * 1024 * 1024`, поэтому в препроцессинге входной снимок ужимается до разрешения `1024 * 1024`, а на постпроцессинге возвращается до исходного.





