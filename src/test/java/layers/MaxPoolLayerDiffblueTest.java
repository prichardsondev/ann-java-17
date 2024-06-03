package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class MaxPoolLayerDiffblueTest {
    /**
     * Method under test: {@link MaxPoolLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 3, 3, 1, 1);

        // Act and Assert
        assertTrue(maxPoolLayer.forwardPass(new ArrayList<>()).isEmpty());
        assertTrue(maxPoolLayer._lastMaxCol.isEmpty());
        assertEquals(maxPoolLayer._lastMaxCol, maxPoolLayer._lastMaxRow);
    }

    /**
     * Method under test: {@link MaxPoolLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass2() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(1, 3, 3, 1, 1);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> maxPoolLayer.forwardPass(input));
    }

    /**
     * Method under test: {@link MaxPoolLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass3() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(2, 3, 3, 1, 1);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act
        List<double[][]> actualForwardPassResult = maxPoolLayer.forwardPass(input);

        // Assert
        assertEquals(1, actualForwardPassResult.size());
        assertEquals(0, actualForwardPassResult.get(0).length);
        assertEquals(1, maxPoolLayer._lastMaxCol.size());
        assertEquals(1, maxPoolLayer._lastMaxRow.size());
    }

    /**
     * Method under test: {@link MaxPoolLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass4() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(0, 3, 3, 1, 1);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(ArithmeticException.class, () -> maxPoolLayer.forwardPass(input));
    }

    /**
     * Method under test: {@link MaxPoolLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass5() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 1, 3, 1, 1);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act
        List<double[][]> actualForwardPassResult = maxPoolLayer.forwardPass(input);

        // Assert
        assertEquals(1, actualForwardPassResult.size());
        assertEquals(1, maxPoolLayer._lastMaxCol.size());
        assertEquals(1, maxPoolLayer._lastMaxRow.size());
        double[][] getResult = actualForwardPassResult.get(0);
        assertEquals(1, getResult.length);
        assertArrayEquals(new double[]{10.0d}, getResult[0], 0.0);
    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(List)}
     */
    @Test
    void testGetOutput() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(1, 3, 3, 1, 1);

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> maxPoolLayer.getOutput(input));
    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(List)}
     */
//    @Test
//    void testGetOutput2() {
//        // Arrange
//        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(0, 3, 3, 1, 1);
//
//        ArrayList<double[][]> input = new ArrayList<>();
//        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});
//
//        // Act
//        maxPoolLayer.getOutput(input);
//
//        // Assert
//        assertTrue(maxPoolLayer._lastMaxCol.isEmpty());
//        assertEquals(maxPoolLayer._lastMaxCol, maxPoolLayer._lastMaxRow);
//    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(List)}
     */
//    @Test
//    void testGetOutput3() {
//        // Arrange
//        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(2, -1, 3, 1, 1);
//        maxPoolLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));
//
//        ArrayList<double[][]> input = new ArrayList<>();
//        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});
//
//        // Act and Assert
//        assertEquals(1, maxPoolLayer._lastMaxCol.size());
//        assertEquals(1, maxPoolLayer._lastMaxRow.size());
//        assertArrayEquals(new double[]{0.0d, 0.0d, 0.0d}, maxPoolLayer.getOutput(input), 0.0);
//    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(List)}
     */
    //    @Test
    //    void testGetOutput2() {
    //        // Arrange
    //        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(0, 3, 3, 1, 1);
    //
    //        ArrayList<double[][]> input = new ArrayList<>();
    //        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});
    //
    //        // Act and Assert
    //        assertThrows(ArithmeticException.class, () -> maxPoolLayer.getOutput(input));
    //    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(List)}
     */
    //    @Test
    //    void testGetOutput3() {
    //        // Arrange
    //        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(2, -1, 3, 1, 1);
    //        maxPoolLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));
    //
    //        ArrayList<double[][]> input = new ArrayList<>();
    //        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});
    //
    //        // Act and Assert
    //        assertEquals(1, maxPoolLayer._lastMaxCol.size());
    //        assertEquals(1, maxPoolLayer._lastMaxRow.size());
    //        assertArrayEquals(new double[]{0.0d, 0.0d, 0.0d}, maxPoolLayer.getOutput(input), 0.0);
    //    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(double[])}
     */
    @Test
    void testGetOutput4() {
        // Arrange, Act and Assert
        assertThrows(NegativeArraySizeException.class,
                () -> (new MaxPoolLayer(1, 3, 3, 1, 1)).getOutput(new double[]{10.0d, 0.5d, 10.0d, 0.5d}));
    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(double[])}
     */
//    @Test
//    void testGetOutput5() {
//        // Arrange
//        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 1, 3, 1, 1);
//        maxPoolLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));
//
//        // Act and Assert
//        assertEquals(3, maxPoolLayer._lastMaxCol.size());
//        assertEquals(3, maxPoolLayer._lastMaxRow.size());
//        assertArrayEquals(new double[]{2.6933439375436112d, 0.0d, 0.0d},
//                maxPoolLayer.getOutput(new double[]{10.0d, 0.5d, 10.0d, 0.5d}), 0.0);
//    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutput(double[])}
     */
    //    @Test
    //    void testGetOutput5() {
    //        // Arrange
    //        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 1, 3, 1, 1);
    //        maxPoolLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));
    //
    //        // Act and Assert
    //        assertEquals(3, maxPoolLayer._lastMaxCol.size());
    //        assertEquals(3, maxPoolLayer._lastMaxRow.size());
    //        assertArrayEquals(new double[]{2.6933439375436112d, 0.0d, 0.0d},
    //                maxPoolLayer.getOutput(new double[]{10.0d, 0.5d, 10.0d, 0.5d}), 0.0);
    //    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 3, 3, 1, 1);

        // Act
        maxPoolLayer.backPropagate(new ArrayList<>());

        // Assert that nothing has changed
        assertNull(maxPoolLayer._lastMaxCol);
        assertNull(maxPoolLayer._lastMaxRow);
        assertNull(maxPoolLayer._nextLayer);
        assertNull(maxPoolLayer._previousLayer);
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate2() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 3, 3, 1, 1);
        MaxPoolLayer _previousLayer = new MaxPoolLayer(3, 3, 3, 1, 1);

        maxPoolLayer.set_previousLayer(_previousLayer);

        // Act
        maxPoolLayer.backPropagate(new ArrayList<>());

        // Assert that nothing has changed
        assertSame(_previousLayer, maxPoolLayer._previousLayer);
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate3() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(0, 3, 3, 1, 1);

        ArrayList<double[][]> dLdO = new ArrayList<>();
        dLdO.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(ArithmeticException.class, () -> maxPoolLayer.backPropagate(dLdO));
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate4() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(1, 3, 3, -1, 1);

        ArrayList<double[][]> dLdO = new ArrayList<>();
        dLdO.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> maxPoolLayer.backPropagate(dLdO));
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate5() {
        // Arrange, Act and Assert
        assertThrows(ArithmeticException.class, () -> (new MaxPoolLayer(0, 3, 3, 1, 1)).backPropagate(new double[]{}));
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate6() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 3, 0, 1, 1);

        // Act
        maxPoolLayer.backPropagate(new double[]{});

        // Assert that nothing has changed
        assertNull(maxPoolLayer._lastMaxCol);
        assertNull(maxPoolLayer._lastMaxRow);
        assertNull(maxPoolLayer._nextLayer);
        assertNull(maxPoolLayer._previousLayer);
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate7() {
        // Arrange, Act and Assert
        assertThrows(NegativeArraySizeException.class,
                () -> (new MaxPoolLayer(3, 3, 3, -1, 1)).backPropagate(new double[]{}));
    }

    /**
     * Method under test: {@link MaxPoolLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate8() {
        // Arrange
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(3, 3, 0, 1, 1);
        MaxPoolLayer _previousLayer = new MaxPoolLayer(3, 3, 3, 1, 1);

        maxPoolLayer.set_previousLayer(_previousLayer);

        // Act
        maxPoolLayer.backPropagate(new double[]{});

        // Assert that nothing has changed
        assertSame(_previousLayer, maxPoolLayer._previousLayer);
    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutputRows()}
     */
    @Test
    void testGetOutputRows() {
        // Arrange, Act and Assert
        assertEquals(1, (new MaxPoolLayer(3, 3, 3, 1, 1)).getOutputRows());
    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutputCols()}
     */
    @Test
    void testGetOutputCols() {
        // Arrange, Act and Assert
        assertEquals(1, (new MaxPoolLayer(3, 3, 3, 1, 1)).getOutputCols());
    }

    /**
     * Method under test: {@link MaxPoolLayer#getOutputElements()}
     */
    @Test
    void testGetOutputElements() {
        // Arrange, Act and Assert
        assertEquals(3, (new MaxPoolLayer(3, 3, 3, 1, 1)).getOutputElements());
    }

    /**
     * Methods under test:
     * <ul>
     *   <li>{@link MaxPoolLayer#MaxPoolLayer(int, int, int, int, int)}
     *   <li>{@link MaxPoolLayer#getOutputLength()}
     * </ul>
     */
    @Test
    void testGettersAndSetters() {
        // Arrange, Act and Assert
        assertEquals(3, (new MaxPoolLayer(3, 3, 3, 1, 1)).getOutputLength());
    }
}
