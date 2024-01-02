# Monkeypox CNN

## The Neural Networks
These convolutional neural networks classify whether or not a skin sample possesses monkeypox based on an image of the skin sample. The models will predict a value close to 0 if the sample is predicted to be of monkeypox and a 1 if the sample is not predicted to be monkeypox. Since all of the models predict binary categorical values, each uses a binary crossentropy loss function and has 1 output neuron. They use standard SGD and Adam optimizers with a learning rate of 0.001 and have dropout layers to prevent overfitting.

1. The first model, found in the **monkeypox_inceptionv3.py** file, is a CNN that uses Tensorflow's **InceptionV3** as a base model (these layers are untrained in the model). It uses an SGD optimizer with an architecture consisting of:
    - 1 Horizontal random flip layer (for image preprocessing)
    - 1 InceptionV3 base model (with an input shape of (128, 128, 3))
    - 1 Flatten layer
    - 1 Dropout layer (with a dropout rate of 0.3)
    - 1 Hidden layer (with 256 neurons and a ReLU activation function)
    - 1 Output layer (with 1 output neuron and a sigmoid activation function)

2. The second model, found in the **monkeypox_vgg16.py** file, uses the pre-trained VGG16 base provided by Tensorflow (these layers are untrained in the model) and only uses a horizontal flip layer to augment the data. It uses an SGD optimizer and has an architecture consisting of:
    - 1 Horizontal random flip layer (for image preprocessing)
    - 1 VGG16 base model (with an input shape of (128, 128, 3))
    - 1 Flatten layer
    - 1 Dropout layer (with a dropout rate of 0.3)
    - 1 Hidden layer (with 256 neurons and a ReLU activation function)
    - 1 Output layer (with 1 output neuron and a sigmoid activation function)

3. The third model, found in the **monkeypox_resenet50v2.py** file, uses the pre-trained ResNet50V2 base provided by Tensorflow (these layers are untrained in the model) and doesn't use any image augmentation techniques. It uses an Adam optimizer and has an architecture consisting of:
    - 1 Horizontal random flip layer (for image preprocessing)
    - 1 Resnet50V2 base model (with an input shape of (128, 128, 3))
    - 1 Global average pooling 2D layer
    - 1 Hidden layer (with 256 neurons and a ReLU activation function)
    - 1 BatchNormalization layer
    - 1 Hidden layer (with 256 neurons and a ReLU activation function)
    - 1 BatchNormalization layer
    - 1 Output layer (with 1 output neuron and a sigmoid activation function)
    
I found that the VGG16 base model tends to get a significantly higher test accuracy than the other two models, but took significantly longer to train. Comparatively, the ResNet50V2 model took less time to train but had a significantly lower test accuracy, while the InceptionV3 algorithm was the fastest at training, but had a slightly lower test accuracy than the ResNet50V2 model. The architecture of the ResNet50V2 model is slightly different because I found it had a higher accuracy with this architecture than the architecture of the other two models. I tried to implement the architecture of the ResNet50V2 model with the other models, but the results among the other models were the same, if not better with the original architecture (the one described above).

Note that when running any of the files, you will need to input the paths of the training, testing, and validation sets as strings — the location for where to put the paths is signified in the file with the words "< PATH TO TRAINING DATA >," "< PATH TO TESTING DATA >," and "< PATH TO VALIDATION DATA >." When you input these paths, they should be such that — when they are concatenated with the individual elements listed in the **path_list** variable — they are complete paths. For example:
> The dataset is stored in a folder called *monkeypox-data*, under which are the respective *train*, *test*, and *valid* directories that can be downloaded from the source (the link to the download site is below)
> - Thus, your file structure is something like:

>     ↓ folder1
>       ↓ folder2
>         ↓ monkeypox-data
>           ↓ train
>             ↓ monkeypox
>                 < Images >
>             ↓ other
>                 < Images >
>           ↓ test
>             ↓ monkeypox
>                 < Images >
>             ↓ other
>                 < Images >
>           ↓ valid
>             ↓ monkeypox
>                 < Images >
>             ↓ other
>                 < Images >

> The paths you input should be something along the lines of: *~/folder1/folder2/monkeypox-data/train/*, *~/folder1/folder2/monkeypox-data/test/*, and *~/folder1/folder2/monkeypox-data/valid/*, and your **path_list** should be set to ['monkeypox', other'], so that when the **create_dataset()** function is running it concatenates the paths with each element in **path_list** to produce fully coherent paths, such as *~/folder1/folder2/monkeypox-data/train/monkeypox*, *~/folder1/folder2/monkeypox-data/train/other*, *~/folder1/folder2/monkeypox-data/test/monkeypox*, etc.
> 

Feel free to further tune the hyperparameters or build upon any of the models!

## The Dataset
The dataset used here can be found at this link: https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset. Credit for the dataset collection goes to **HASIB AL MUZDADID**, **stpete_ishii**, **Gerry**, and others on *Kaggle*. The dataset used by these models (found in the Fold1 folder on the website above) contains approximately 2142 training images, 45 testing images, and 420 validation images. Note that the images from the original dataset are resized to 128 x 128 images so that they are more manageable for the model. The dataset is not included in the repository because it is too large to stabley upload to Github, so just use the link above to find and download the dataset.

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install

## Disclaimer
Please note that I do not recommend, endorse, or encourage the use of any of my work here in actual medical use or application in any way. 
