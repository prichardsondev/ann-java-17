package network;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import layers.Layer;
import org.junit.jupiter.api.Test;

class NetworkBuilderDiffblueTest {
    /**
     * Method under test:
     * {@link NetworkBuilder#addConvolutionLayer(int, int, int, double, long)}
     */
    @Test
    void testAddConvolutionLayer() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);

        // Act
        networkBuilder.addConvolutionLayer(10, 3, 3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(1, layerList.size());
        Layer getResult = layerList.get(0);
        assertEquals(1, getResult.getOutputCols());
        assertEquals(1, getResult.getOutputRows());
        assertEquals(10, getResult.getOutputLength());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addConvolutionLayer(int, int, int, double, long)}
     */
    @Test
    void testAddConvolutionLayer2() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);
        networkBuilder.addConvolutionLayer(10, 3, 3, 10.0d, 42L);

        // Act
        networkBuilder.addConvolutionLayer(10, 3, 3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(2, layerList.size());
        Layer getResult = layerList.get(1);
        assertEquals(1, getResult.getOutputCols());
        assertEquals(1, getResult.getOutputRows());
        assertEquals(100, getResult.getOutputLength());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addConvolutionLayer(int, int, int, double, long)}
     */
    @Test
    void testAddConvolutionLayer3() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);
        networkBuilder.addMaxPoolLayer(3, 3);

        // Act
        networkBuilder.addConvolutionLayer(10, 3, 3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(2, layerList.size());
        Layer getResult = layerList.get(1);
        assertEquals(1, getResult.getOutputCols());
        assertEquals(1, getResult.getOutputRows());
        assertEquals(10, getResult.getOutputLength());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addConvolutionLayer(int, int, int, double, long)}
     */
    @Test
    void testAddConvolutionLayer4() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);
        networkBuilder.addFullyConnectedLayer(3, 10.0d, 42L);

        // Act
        networkBuilder.addConvolutionLayer(10, 3, 3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(2, layerList.size());
        Layer getResult = layerList.get(1);
        assertEquals(0, getResult.getOutputCols());
        assertEquals(0, getResult.getOutputLength());
        assertEquals(0, getResult.getOutputRows());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addFullyConnectedLayer(int, double, long)}
     */
    @Test
    void testAddFullyConnectedLayer() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);

        // Act
        networkBuilder.addFullyConnectedLayer(3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(1, layerList.size());
        assertEquals(3, layerList.get(0).getOutputElements());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addFullyConnectedLayer(int, double, long)}
     */
    @Test
    void testAddFullyConnectedLayer2() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);
        networkBuilder.addConvolutionLayer(10, 3, 3, 10.0d, 42L);

        // Act
        networkBuilder.addFullyConnectedLayer(3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(2, layerList.size());
        assertEquals(3, layerList.get(1).getOutputElements());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addFullyConnectedLayer(int, double, long)}
     */
    @Test
    void testAddFullyConnectedLayer3() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);
        networkBuilder.addMaxPoolLayer(3, 3);

        // Act
        networkBuilder.addFullyConnectedLayer(3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(2, layerList.size());
        assertEquals(3, layerList.get(1).getOutputElements());
    }

    /**
     * Method under test:
     * {@link NetworkBuilder#addFullyConnectedLayer(int, double, long)}
     */
    @Test
    void testAddFullyConnectedLayer4() {
        // Arrange
        NetworkBuilder networkBuilder = new NetworkBuilder(1, 1, 10.0d);
        networkBuilder.addFullyConnectedLayer(3, 10.0d, 42L);

        // Act
        networkBuilder.addFullyConnectedLayer(3, 10.0d, 42L);

        // Assert
        List<Layer> layerList = networkBuilder._layers;
        assertEquals(2, layerList.size());
        assertEquals(3, layerList.get(1).getOutputElements());
    }

    /**
     * Methods under test:
     * <ul>
     *   <li>{@link NetworkBuilder#build()}
     *   <li>{@link NetworkBuilder#NetworkBuilder(int, int, double)}
     * </ul>
     */
    @Test
    void testBuild() {
        // Arrange, Act and Assert
        assertTrue((new NetworkBuilder(1, 1, 10.0d)).build()._layers.isEmpty());
    }
}
