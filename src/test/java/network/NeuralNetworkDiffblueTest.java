package network;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import data.Image;

import java.util.ArrayList;
import java.util.List;

import layers.ConvolutionLayer;
import layers.FullyConnectedLayer;
import layers.Layer;
import org.junit.jupiter.api.Test;

class NeuralNetworkDiffblueTest {
    /**
     * Method under test: {@link NeuralNetwork#getErrors(double[], int)}
     */
    @Test
    void testGetErrors() {
        // Arrange, Act and Assert
        assertArrayEquals(new double[]{10.0d, 1.0d, 10.0d, 2.0d},
                (new NeuralNetwork(new ArrayList<>(), 10.0d)).getErrors(new double[]{10.0d, 2.0d, 10.0d, 2.0d}, 1), 0.0);
    }

    /**
     * Method under test: {@link NeuralNetwork#guess(Image)}
     */
    @Test
    void testGuess() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 3, 42L, 1.0d));
        NeuralNetwork neuralNetwork = new NeuralNetwork(layers, 10.0d);

        // Act and Assert
        assertEquals(0, neuralNetwork.guess(new Image(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}, 1)));
    }

    /**
     * Method under test: {@link NeuralNetwork#guess(Image)}
     */
    @Test
    void testGuess2() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 2, 42L, 1.0d));
        NeuralNetwork neuralNetwork = new NeuralNetwork(layers, 10.0d);

        // Act and Assert
        assertEquals(1, neuralNetwork.guess(new Image(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}, 1)));
    }

    /**
     * Method under test: {@link NeuralNetwork#guess(Image)}
     */
    @Test
    void testGuess3() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 1.0d);
        convolutionLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 1.0d));

        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 3, 42L, 1.0d));
        layers.add(convolutionLayer);
        NeuralNetwork neuralNetwork = new NeuralNetwork(layers, 10.0d);

        // Act and Assert
        assertEquals(0, neuralNetwork.guess(new Image(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}, 1)));
    }

    /**
     * Method under test: {@link NeuralNetwork#test(List)}
     */
    @Test
    void testTest() {
        // Arrange
        NeuralNetwork neuralNetwork = new NeuralNetwork(new ArrayList<>(), 10.0d);

        // Act and Assert
        assertEquals(Float.NaN, neuralNetwork.test(new ArrayList<>()));
    }

    /**
     * Method under test: {@link NeuralNetwork#test(List)}
     */
    @Test
    void testTest2() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 3, 42L, 1.0d));
        NeuralNetwork neuralNetwork = new NeuralNetwork(layers, 10.0d);

        ArrayList<Image> images = new ArrayList<>();
        images.add(new Image(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}, 1));

        // Act and Assert
        assertEquals(0.0f, neuralNetwork.test(images));
    }

    /**
     * Method under test: {@link NeuralNetwork#test(List)}
     */
    @Test
    void testTest3() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 2, 42L, 1.0d));
        NeuralNetwork neuralNetwork = new NeuralNetwork(layers, 10.0d);

        ArrayList<Image> images = new ArrayList<>();
        images.add(new Image(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}, 1));

        // Act and Assert
        assertEquals(1.0f, neuralNetwork.test(images));
    }

    /**
     * Method under test: {@link NeuralNetwork#train(List)}
     */
    @Test
    void testTrain() {
        // Arrange
        NeuralNetwork neuralNetwork = new NeuralNetwork(new ArrayList<>(), 10.0d);

        // Act
        neuralNetwork.train(new ArrayList<>());

        // Assert that nothing has changed
        assertEquals(10.0d, neuralNetwork._scaleFactor);
        assertTrue(neuralNetwork._layers.isEmpty());
    }

    /**
     * Method under test: {@link NeuralNetwork#NeuralNetwork(List, double)}
     */
    @Test
    void testNewNeuralNetwork() {
        // Arrange and Act
        NeuralNetwork actualNeuralNetwork = new NeuralNetwork(new ArrayList<>(), 10.0d);

        // Assert
        assertEquals(10.0d, actualNeuralNetwork._scaleFactor);
        assertTrue(actualNeuralNetwork._layers.isEmpty());
    }

    /**
     * Method under test: {@link NeuralNetwork#NeuralNetwork(List, double)}
     */
    @Test
    void testNewNeuralNetwork2() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        // Act
        NeuralNetwork actualNeuralNetwork = new NeuralNetwork(layers, 10.0d);

        // Assert
        assertEquals(1, actualNeuralNetwork._layers.size());
        assertEquals(10.0d, actualNeuralNetwork._scaleFactor);
    }

    /**
     * Method under test: {@link NeuralNetwork#NeuralNetwork(List, double)}
     */
    @Test
    void testNewNeuralNetwork3() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 3, 42L, 10.0d));
        layers.add(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        // Act
        NeuralNetwork actualNeuralNetwork = new NeuralNetwork(layers, 10.0d);

        // Assert
        assertEquals(10.0d, actualNeuralNetwork._scaleFactor);
        assertEquals(2, actualNeuralNetwork._layers.size());
    }

    /**
     * Method under test: {@link NeuralNetwork#NeuralNetwork(List, double)}
     */
    @Test
    void testNewNeuralNetwork4() {
        // Arrange
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new FullyConnectedLayer(3, 3, 42L, 10.0d));
        layers.add(new FullyConnectedLayer(3, 3, 42L, 10.0d));
        layers.add(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        // Act
        NeuralNetwork actualNeuralNetwork = new NeuralNetwork(layers, 10.0d);

        // Assert
        assertEquals(10.0d, actualNeuralNetwork._scaleFactor);
        assertEquals(3, actualNeuralNetwork._layers.size());
    }
}
