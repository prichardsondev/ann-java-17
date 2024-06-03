Code developed by following tutorial here - and then 4 hours of looking for my errors.  
https://www.youtube.com/watch?v=3MMonOWGe0M&list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN&index=1  

Add mnist_test.csv and mnist_test.csv to ./data folder
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv  

#### main method

We first runs a test before any training Epochs (fancy word for run) - will be low percent success rate. After that  
each epoch will train your model and test. Slow on my machine - about 5 minutes per Epoch.  

Sample output:  
Loading data....  
Test data size = 10000  
Training data size = 60000  
Training success rate: 0.0739  
Epoch 0  
Training success rate: 0.8625  
Epoch 1  
Training success rate: 0.8982  
Epoch 2  
Training success rate: 0.9091  

Process finished with exit code 0  

We can comment out the Convolution and MaxPool layers, so they are not added to our network builder.  
Running just the fully connected layer it takes 69 Epochs to reach the same success rate as 3 epochs above, 
but it runs in a fraction of the time.  Note this will not translate as well when running inference against unknown data.

...  
Epoch 69
Training success rate: 0.9094

#### Convolutional Layers:

Convolutional layers apply learnable filters to input images, extracting features such as edges, textures, and patterns. By scanning the entire image with these filters, convolutional layers capture spatial hierarchies of features, enabling the network to learn representations robust to variations in position and scale.

#### Max-Pooling Layers:

Max-pooling layers follow convolutional layers, reducing the spatial dimensions of feature maps. They downsample the feature maps by taking the maximum value from small, overlapping regions. Max-pooling helps improve computational efficiency, reduce overfitting, and retain salient features by summarizing the most important information.

#### Fully Connected Layers:

Fully connected layers receive flattened feature vectors from the preceding layers and perform classification based on these features. While convolutional and max-pooling layers extract hierarchical representations of features, fully connected layers make the final decision about class probabilities. They integrate information from all features to classify input data accurately.

#### Working Together:

Convolutional layers extract meaningful features from input images, while max-pooling layers reduce spatial dimensions and retain important information. Together, they enable the network to learn hierarchical representations of features efficiently. Fully connected layers then use these representations for classification, making the final decision about the input's class probabilities.

