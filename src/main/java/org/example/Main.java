package org.example;

import data.DataReader;
import data.Image;
import network.NetworkBuilder;
import network.NeuralNetwork;

import java.util.List;

import static java.util.Collections.shuffle;

/**
 * The main method loads data, builds a neural network, and trains it on the MNIST dataset.
 * It consists of the following steps:
 * 1. Loading MNIST test and training data.
 * 2. Building a neural network with a convolutional layer, max-pooling layer, and fully connected layer.
 *    - Convolution Layer: Extracts features from the input images using convolution operations.
 *    - MaxPooling Layer: Reduces the spatial dimensions of the feature maps while retaining important information.
 *    - Fully Connected Layer: Performs classification based on the extracted features.
 * 3. Testing the network's performance on the test dataset before training.
 * 4. Training the network for a specified number of epochs using the training dataset.
 * 5. Testing the network's performance on the test dataset after each epoch of training.
 *
 * @param args The command-line arguments (not used).
 */

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        long seed = 123;

        System.out.println("Loading data....");
        List<Image> test = DataReader.readData("data/mnist_test.csv");
        List<Image> training = DataReader.readData("data/mnist_train.csv");

        System.out.println("Test data size = " + test.size());
        System.out.println("Training data size = " + training.size());

        NetworkBuilder builder = new NetworkBuilder(28,28, 256*100);
        builder.addConvolutionLayer(8,5,1,0.1, seed);
        builder.addMaxPoolLayer(3,2);
        builder.addFullyConnectedLayer(10,0.1 ,seed);

        NeuralNetwork net = builder.build();

        float rate = net.test(test);
        System.out.println("Training success rate: " + rate);

        int epochs = 3;

        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch " + i);
            shuffle(training);
            net.train(training);
            rate = net.test(test);
            System.out.println("Training success rate: " + rate);
        }

    }
}