### RVL-CDIP Image Classification

This repository contains code and models for the RVL-CDIP image classification task using an ensemble of pre-trained CNNs and transformers. The goal is to classify 16000 grayscale document images into one of the 16 classes or document types.

### Dataset

The RVL-CDIP dataset is used for training and validation. It consists of 16,000 images, with approximately 1,000 images per class. The validation dataset contains 900 images.

### Experimental Settings

The experiments were conducted on Google Colab using a standard K80 GPU and 12 GB of RAM. The specific libraries used for different models are mentioned in the Requirements.pdf file.

Download the Kaggle Dataset for this problem given in this [link](https://www.kaggle.com/competitions/datathonindoml-2022/data).

After downloading the .zip file of the dataset from kaggle, open google drive and create a seperate folder, Datathon and upload the .zip file of dataset in there.

Now open the colab notebook, mount your google drive from the sidebar, connect with a runtime GPU and start execution of the first cell which will unzip the files from the .zip dataset.

After unzipping, you will get 4 things, training and validation folders, sample-submission.csv , train-labels.csv

Now you are ready with the initial configuration. You can now start running the notebook.

### Data Preprocessing

    Dealing with TIF Images: The TIF files are renamed to JPEG files using the rename() function from the os module because TensorFlow does not support TIF files.
    
    Resizing the Images: The image files are converted to RGB format using the convert() function from the Python Imaging Library (PIL) to ensure they have three color channels and are resized to 224x224 pixels, which is the input size required by the model.
    
    Converting into Tensors: The images are transformed into tensors by first converting them into NumPy arrays and then into tensors using the constant() function from TensorFlow. This allows them to be fed into the model.

    Creating Batches: Data batches of size 32 are created using the batch() function from the TensorFlow library. Loading the images in batches requires less memory compared to loading the entire dataset. Mini Gradient Descent is used for training, which is faster than both batch or stochastic gradient descent.

### Method Description & Model Architecture

The models used are ResNet, DenseNet, EfficientNet, MobileNet, Long Short-Term Memory (LSTM), few shot, Ensemble Models and Vision Transformer.

### Training and Evaluation

The models are trained using the RVL-CDIP dataset. The training process involves feeding batches of preprocessed images into the models and updating the model's parameters using backpropagation and gradient descent optimization. The models are evaluated on the validation dataset, and metrics such as accuracy, precision, recall, and F1 score are computed to assess their performance.
