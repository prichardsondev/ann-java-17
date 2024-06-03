package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class FullyConnectedLayerDiffblueTest {
    /**
     * Method under test: {@link FullyConnectedLayer#forwardPass(double[])}
     */
    @Test
    void testForwardPass() {
        // Arrange, Act and Assert
        assertArrayEquals(new double[]{3.235769177879286d, 0.0d, 0.0d},
                (new FullyConnectedLayer(3, 3, 42L, 10.0d)).forwardPass(new double[]{10.0d, 0.01d, 10.0d, 0.01d}), 0.0);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#getOutput(List)}
     */
    @Test
    void testGetOutput() {
        // Arrange
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(3, 3, 42L, 10.0d);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertArrayEquals(new double[]{2.6933439375436112d, 0.0d, 0.0d}, fullyConnectedLayer.getOutput(input), 0.0);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#getOutput(List)}
     */
    @Test
    void testGetOutput2() {
        // Arrange
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(3, 3, 42L, 10.0d);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{
                new double[]{2.6933439375436112d, -4.631865717916165d, 2.6933439375436112d, -4.631865717916165d}});
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertArrayEquals(new double[]{6.001915793069248d, 0.0d, 0.0d}, fullyConnectedLayer.getOutput(input), 0.0);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#getOutput(List)}
     */
    @Test
    void testGetOutput3() {
        // Arrange
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(3, 3, 42L, 10.0d);
        fullyConnectedLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 2.6933439375436112d));

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertArrayEquals(new double[]{3.075543758678177d, 2.4762818255221974d, 0.0d}, fullyConnectedLayer.getOutput(input),
                0.0);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#getOutput(double[])}
     */
    @Test
    void testGetOutput4() {
        // Arrange, Act and Assert
        assertArrayEquals(new double[]{3.235769177879286d, 0.0d, 0.0d},
                (new FullyConnectedLayer(3, 3, 42L, 10.0d)).getOutput(new double[]{10.0d, 0.01d, 10.0d, 0.01d}), 0.0);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#getOutput(double[])}
     */
    @Test
    void testGetOutput5() {
        // Arrange
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(3, 3, 42L, 10.0d);
        fullyConnectedLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 3.235769177879286d));

        // Act and Assert
        assertArrayEquals(new double[]{3.694942023864233d, 2.974991903215716d, 0.0d},
                fullyConnectedLayer.getOutput(new double[]{10.0d, 0.01d, 10.0d, 0.01d}), 0.0);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate() {
        // Arrange
        ConvolutionLayer _previousLayer = new ConvolutionLayer(3, 3, 0, 3, 3, 42L, 10, 10.0d);
        _previousLayer.forwardPass(new ArrayList<>());

        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(0, 3, 42L, 10.0d);
        fullyConnectedLayer.set_previousLayer(_previousLayer);

        ArrayList<double[][]> dLdO = new ArrayList<>();
        dLdO.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act
        fullyConnectedLayer.backPropagate(dLdO);

        // Assert
        assertEquals(0, fullyConnectedLayer._previousLayer.getOutputElements());
    }

    /**
     * Method under test: {@link FullyConnectedLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate2() {
        // Arrange
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(0, 3, 42L, 10.0d);

        // Act
        fullyConnectedLayer.backPropagate(new double[]{10.0d, 0.01d, 10.0d, 0.01d});

        // Assert that nothing has changed
        assertNull(fullyConnectedLayer._nextLayer);
        assertNull(fullyConnectedLayer._previousLayer);
    }

    /**
     * Method under test: {@link FullyConnectedLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate3() {
        // Arrange
        ConvolutionLayer _previousLayer = new ConvolutionLayer(3, 3, 0, 3, 3, 42L, 10, 10.0d);
        _previousLayer.forwardPass(new ArrayList<>());

        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(0, 3, 42L, 10.0d);
        fullyConnectedLayer.set_previousLayer(_previousLayer);

        // Act
        fullyConnectedLayer.backPropagate(new double[]{10.0d, 0.01d, 10.0d, 0.01d});

        // Assert
        assertEquals(0, fullyConnectedLayer._previousLayer.getOutputElements());
    }

    /**
     * Method under test: {@link FullyConnectedLayer#setRandomWeights()}
     */
    @Test
    void testSetRandomWeights() {
        // TODO: Diffblue Cover was only able to create a partial test for this method:
        //   Diffblue AI was unable to find a test

        // Arrange and Act
        (new FullyConnectedLayer(3, 3, 42L, 10.0d)).setRandomWeights();
    }

    /**
     * Method under test: {@link FullyConnectedLayer#relu(double)}
     */
    @Test
    void testRelu() {
        // Arrange, Act and Assert
        assertEquals(10.0d, (new FullyConnectedLayer(3, 3, 42L, 10.0d)).relu(10.0d));
        assertEquals(0.0d, (new FullyConnectedLayer(3, 3, 42L, 10.0d)).relu(0.0d));
    }

    /**
     * Method under test: {@link FullyConnectedLayer#derivativeReLu(double)}
     */
    @Test
    void testDerivativeReLu() {
        // Arrange, Act and Assert
        assertEquals(1.0d, (new FullyConnectedLayer(3, 3, 42L, 10.0d)).derivativeReLu(10.0d));
        assertEquals(0.01d, (new FullyConnectedLayer(3, 3, 42L, 10.0d)).derivativeReLu(0.0d));
    }

    /**
     * Methods under test:
     * <ul>
     *   <li>{@link FullyConnectedLayer#getOutputCols()}
     *   <li>{@link FullyConnectedLayer#getOutputElements()}
     *   <li>{@link FullyConnectedLayer#getOutputLength()}
     *   <li>{@link FullyConnectedLayer#getOutputRows()}
     * </ul>
     */
    @Test
    void testGettersAndSetters() {
        // Arrange
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(3, 3, 42L, 10.0d);

        // Act
        int actualOutputCols = fullyConnectedLayer.getOutputCols();
        int actualOutputElements = fullyConnectedLayer.getOutputElements();
        int actualOutputLength = fullyConnectedLayer.getOutputLength();

        // Assert
        assertEquals(0, actualOutputCols);
        assertEquals(0, actualOutputLength);
        assertEquals(0, fullyConnectedLayer.getOutputRows());
        assertEquals(3, actualOutputElements);
    }

    /**
     * Method under test:
     * {@link FullyConnectedLayer#FullyConnectedLayer(int, int, long, double)}
     */
    @Test
    void testNewFullyConnectedLayer() {
        // Arrange, Act and Assert
        assertEquals(3, (new FullyConnectedLayer(3, 3, 42L, 10.0d)).getOutputElements());
        assertThrows(NegativeArraySizeException.class, () -> new FullyConnectedLayer(-1, 3, 42L, 10.0d));
    }
}
