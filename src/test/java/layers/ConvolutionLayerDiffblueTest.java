package layers;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class ConvolutionLayerDiffblueTest {
    /**
     * Method under test: {@link ConvolutionLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);

        // Act and Assert
        assertTrue(convolutionLayer.forwardPass(new ArrayList<>()).isEmpty());
    }

    /**
     * Method under test: {@link ConvolutionLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass2() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);

        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act
        List<double[][]> actualForwardPassResult = convolutionLayer.forwardPass(list);

        // Assert
        assertEquals(10, actualForwardPassResult.size());
        double[][] getResult = actualForwardPassResult.get(0);
        assertEquals(1, getResult.length);
        double[][] getResult2 = actualForwardPassResult.get(1);
        assertEquals(1, getResult2.length);
        double[][] getResult3 = actualForwardPassResult.get(2);
        assertEquals(1, getResult3.length);
        double[][] getResult4 = actualForwardPassResult.get(3);
        assertEquals(1, getResult4.length);
        double[][] getResult5 = actualForwardPassResult.get(4);
        assertEquals(1, getResult5.length);
        double[][] getResult6 = actualForwardPassResult.get(5);
        assertEquals(1, getResult6.length);
        double[][] getResult7 = actualForwardPassResult.get(6);
        assertEquals(1, getResult7.length);
        double[][] getResult8 = actualForwardPassResult.get(7);
        assertEquals(1, getResult8.length);
        double[][] getResult9 = actualForwardPassResult.get(8);
        assertEquals(1, getResult9.length);
        double[][] getResult10 = actualForwardPassResult.get(9);
        assertEquals(1, getResult10.length);
        assertEquals(1, (getResult[0]).length);
        assertEquals(1, (getResult2[0]).length);
        assertEquals(1, (getResult3[0]).length);
        assertEquals(1, (getResult4[0]).length);
        assertEquals(1, (getResult5[0]).length);
        assertEquals(1, (getResult6[0]).length);
        assertEquals(1, (getResult7[0]).length);
        assertEquals(1, (getResult8[0]).length);
        assertEquals(1, (getResult9[0]).length);
        assertArrayEquals(new double[]{0.0d}, getResult10[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass3() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);

        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act
        List<double[][]> actualForwardPassResult = convolutionLayer.forwardPass(list);

        // Assert
        assertEquals(20, actualForwardPassResult.size());
        double[][] getResult = actualForwardPassResult.get(0);
        assertEquals(1, getResult.length);
        double[][] getResult2 = actualForwardPassResult.get(1);
        assertEquals(1, getResult2.length);
        double[][] getResult3 = actualForwardPassResult.get(14);
        assertEquals(1, getResult3.length);
        double[][] getResult4 = actualForwardPassResult.get(15);
        assertEquals(1, getResult4.length);
        double[][] getResult5 = actualForwardPassResult.get(17);
        assertEquals(1, getResult5.length);
        double[][] getResult6 = actualForwardPassResult.get(18);
        assertEquals(1, getResult6.length);
        double[][] getResult7 = actualForwardPassResult.get(19);
        assertEquals(1, getResult7.length);
        double[][] getResult8 = actualForwardPassResult.get(2);
        assertEquals(1, getResult8.length);
        double[][] getResult9 = actualForwardPassResult.get(3);
        assertEquals(1, getResult9.length);
        double[][] getResult10 = actualForwardPassResult.get(4);
        assertEquals(1, getResult10.length);
        double[][] getResult11 = actualForwardPassResult.get(5);
        assertEquals(1, getResult11.length);
        double[][] getResult12 = actualForwardPassResult.get(Short.SIZE);
        assertEquals(1, getResult12.length);
        assertEquals(1, (getResult[0]).length);
        assertEquals(1, (getResult2[0]).length);
        assertEquals(1, (getResult3[0]).length);
        assertEquals(1, (getResult4[0]).length);
        assertEquals(1, (getResult5[0]).length);
        assertEquals(1, (getResult6[0]).length);
        assertEquals(1, (getResult8[0]).length);
        assertEquals(1, (getResult9[0]).length);
        assertEquals(1, (getResult10[0]).length);
        assertEquals(1, (getResult11[0]).length);
        assertEquals(1, (getResult12[0]).length);
        assertArrayEquals(new double[]{0.0d}, getResult7[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass4() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 3, 3, 1, 1, 42L, 10, 10.0d);

        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act
        List<double[][]> actualForwardPassResult = convolutionLayer.forwardPass(list);

        // Assert
        assertEquals(10, actualForwardPassResult.size());
        double[][] getResult = actualForwardPassResult.get(0);
        assertEquals(1, getResult.length);
        double[][] getResult2 = actualForwardPassResult.get(1);
        assertEquals(1, getResult2.length);
        double[][] getResult3 = actualForwardPassResult.get(2);
        assertEquals(1, getResult3.length);
        double[][] getResult4 = actualForwardPassResult.get(3);
        assertEquals(1, getResult4.length);
        double[][] getResult5 = actualForwardPassResult.get(4);
        assertEquals(1, getResult5.length);
        double[][] getResult6 = actualForwardPassResult.get(5);
        assertEquals(1, getResult6.length);
        double[][] getResult7 = actualForwardPassResult.get(6);
        assertEquals(1, getResult7.length);
        double[][] getResult8 = actualForwardPassResult.get(7);
        assertEquals(1, getResult8.length);
        double[][] getResult9 = actualForwardPassResult.get(8);
        assertEquals(1, getResult9.length);
        double[][] getResult10 = actualForwardPassResult.get(9);
        assertEquals(1, getResult10.length);
        assertEquals(2, (getResult[0]).length);
        assertEquals(2, (getResult2[0]).length);
        assertEquals(2, (getResult3[0]).length);
        assertEquals(2, (getResult4[0]).length);
        assertEquals(2, (getResult5[0]).length);
        assertEquals(2, (getResult6[0]).length);
        assertEquals(2, (getResult7[0]).length);
        assertEquals(2, (getResult8[0]).length);
        assertEquals(2, (getResult9[0]).length);
        assertArrayEquals(new double[]{14.862133923906502d, 0.7431066961953251d}, getResult10[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#forwardPass(List)}
     */
    @Test
    void testForwardPass5() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(7, 3, 3, 1, 1, 42L, 10, 10.0d);

        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> convolutionLayer.forwardPass(list));
    }

    /**
     * Method under test: {@link ConvolutionLayer#spaceArray(double[][])}
     */
    @Test
    void testSpaceArray() {
        // Arrange and Act
        double[][] actualSpaceArrayResult = (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d))
                .spaceArray(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualSpaceArrayResult.length);
        assertArrayEquals(new double[]{10.0d, 0.0d, 0.0d, 0.5d, 0.0d, 0.0d, 10.0d, 0.0d, 0.0d, 0.5d},
                actualSpaceArrayResult[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#spaceArray(double[][])}
     */
    @Test
    void testSpaceArray2() {
        // Arrange and Act
        double[][] actualSpaceArrayResult = (new ConvolutionLayer(3, 1, 3, 1, 1, 42L, 10, 10.0d))
                .spaceArray(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualSpaceArrayResult.length);
        assertArrayEquals(new double[]{10.0d, 0.5d, 10.0d, 0.5d}, actualSpaceArrayResult[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#spaceArray(double[][])}
     */
    @Test
    void testSpaceArray3() {
        // Arrange, Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> (new ConvolutionLayer(3, -1, 3, 1, 1, 42L, 10, 10.0d))
                .spaceArray(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}));
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutput(List)}
     */
    @Test
    void testGetOutput() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertArrayEquals(new double[]{0.0d, 0.0d, 0.0d}, convolutionLayer.getOutput(input), 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutput(List)}
     */
    @Test
    void testGetOutput2() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(7, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        ArrayList<double[][]> input = new ArrayList<>();
        input.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> convolutionLayer.getOutput(input));
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutput(double[])}
     */
    @Test
    void testGetOutput3() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        // Act and Assert
        assertArrayEquals(new double[]{0.0d, 0.0d, 0.0d},
                convolutionLayer.getOutput(new double[]{10.0d, 0.5d, 10.0d, 0.5d}), 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutput(double[])}
     */
    @Test
    void testGetOutput4() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        // Act and Assert
        assertArrayEquals(new double[]{10.624234305844062d, 26.348348697997974d, 0.0d},
                convolutionLayer.getOutput(new double[]{10.0d, 0.5d, 10.0d, 0.5d}), 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutput(double[])}
     */
    @Test
    void testGetOutput5() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(7, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.set_nextLayer(new FullyConnectedLayer(3, 3, 42L, 10.0d));

        // Act and Assert
        assertThrows(NegativeArraySizeException.class,
                () -> convolutionLayer.getOutput(new double[]{10.0d, 0.5d, 10.0d, 0.5d}));
    }

    /**
     * Method under test: {@link ConvolutionLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.forwardPass(new ArrayList<>());

        // Act
        convolutionLayer.backPropagate(new ArrayList<>());

        // Assert
        assertEquals(30, convolutionLayer.getOutputElements());
    }

    /**
     * Method under test: {@link ConvolutionLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate2() {
        // Arrange
        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, -1, 1, 42L, 10, 10.0d);
        convolutionLayer.forwardPass(list);

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> convolutionLayer.backPropagate(new ArrayList<>()));
    }

    /**
     * Method under test: {@link ConvolutionLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate3() {
        // Arrange
        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.forwardPass(list);

        ArrayList<double[][]> dLdO = new ArrayList<>();
        dLdO.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> convolutionLayer.backPropagate(dLdO));
    }

    /**
     * Method under test: {@link ConvolutionLayer#backPropagate(List)}
     */
    @Test
    void testBackPropagate4() {
        // Arrange
        ArrayList<double[][]> list = new ArrayList<>();
        list.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, -1, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.forwardPass(list);

        ArrayList<double[][]> dLdO = new ArrayList<>();
        dLdO.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> convolutionLayer.backPropagate(dLdO));
    }

    /**
     * Method under test: {@link ConvolutionLayer#backPropagate(double[])}
     */
    @Test
    void testBackPropagate5() {
        // Arrange
        ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);
        convolutionLayer.forwardPass(new ArrayList<>());

        // Act
        convolutionLayer.backPropagate(new double[]{10.0d, 0.5d, 10.0d, 0.5d});

        // Assert
        assertEquals(30, convolutionLayer.getOutputElements());
    }

    /**
     * Method under test: {@link ConvolutionLayer#flipArrayHorizontally(double[][])}
     */
    @Test
    void testFlipArrayHorizontally() {
        // Arrange and Act
        double[][] actualFlipArrayHorizontallyResult = (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d))
                .flipArrayHorizontally(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualFlipArrayHorizontallyResult.length);
        assertArrayEquals(new double[]{0.5d, 10.0d, 0.5d, 10.0d}, actualFlipArrayHorizontallyResult[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#flipArrayVertically(double[][])}
     */
    @Test
    void testFlipArrayVertically() {
        // Arrange and Act
        double[][] actualFlipArrayVerticallyResult = (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d))
                .flipArrayVertically(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualFlipArrayVerticallyResult.length);
        assertArrayEquals(new double[]{10.0d, 0.5d, 10.0d, 0.5d}, actualFlipArrayVerticallyResult[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#flipArrayDiagonally(double[][])}
     */
    @Test
    void testFlipArrayDiagonally() {
        // Arrange and Act
        double[][] actualFlipArrayDiagonallyResult = (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d))
                .flipArrayDiagonally(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualFlipArrayDiagonallyResult.length);
        assertArrayEquals(new double[]{0.5d, 10.0d, 0.5d, 10.0d}, actualFlipArrayDiagonallyResult[0], 0.0);
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutputLength()}
     */
    @Test
    void testGetOutputLength() {
        // Arrange, Act and Assert
        assertEquals(30, (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d)).getOutputLength());
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutputRows()}
     */
    @Test
    void testGetOutputRows() {
        // Arrange, Act and Assert
        assertEquals(1, (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d)).getOutputRows());
        assertThrows(ArithmeticException.class,
                () -> (new ConvolutionLayer(3, 0, 3, 1, 1, 42L, 10, 10.0d)).getOutputRows());
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutputCols()}
     */
    @Test
    void testGetOutputCols() {
        // Arrange, Act and Assert
        assertEquals(1, (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d)).getOutputCols());
        assertThrows(ArithmeticException.class,
                () -> (new ConvolutionLayer(3, 0, 3, 1, 1, 42L, 10, 10.0d)).getOutputCols());
    }

    /**
     * Method under test: {@link ConvolutionLayer#getOutputElements()}
     */
    @Test
    void testGetOutputElements() {
        // Arrange, Act and Assert
        assertEquals(30, (new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d)).getOutputElements());
        assertThrows(ArithmeticException.class,
                () -> (new ConvolutionLayer(3, 0, 3, 1, 1, 42L, 10, 10.0d)).getOutputElements());
    }

    /**
     * Method under test:
     * {@link ConvolutionLayer#ConvolutionLayer(int, int, int, int, int, long, int, double)}
     */
    @Test
    void testNewConvolutionLayer() {
        // Arrange and Act
        ConvolutionLayer actualConvolutionLayer = new ConvolutionLayer(3, 3, 3, 1, 1, 42L, 10, 10.0d);

        // Assert
        assertEquals(1, actualConvolutionLayer.getOutputCols());
        assertEquals(1, actualConvolutionLayer.getOutputRows());
        assertEquals(30, actualConvolutionLayer.getOutputLength());
    }

    /**
     * Method under test:
     * {@link ConvolutionLayer#ConvolutionLayer(int, int, int, int, int, long, int, double)}
     */
    @Test
    void testNewConvolutionLayer2() {
        // Arrange, Act and Assert
        assertThrows(NegativeArraySizeException.class, () -> new ConvolutionLayer(-1, 3, 3, 1, 1, 42L, 10, 10.0d));

    }
}
