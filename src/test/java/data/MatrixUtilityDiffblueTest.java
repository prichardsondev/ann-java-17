package data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class MatrixUtilityDiffblueTest {
    /**
     * Method under test: {@link MatrixUtility#add(double[], double[])}
     */
    @Test
    void testAdd() {
        // Arrange, Act and Assert
        assertArrayEquals(new double[]{20.0d, 1.0d, 20.0d, 1.0d},
                MatrixUtility.add(new double[]{10.0d, 0.5d, 10.0d, 0.5d}, new double[]{10.0d, 0.5d, 10.0d, 0.5d}), 0.0);
    }

    /**
     * Method under test: {@link MatrixUtility#add(double[][], double[][])}
     */
    @Test
    void testAdd2() {
        // Arrange and Act
        double[][] actualAddResult = MatrixUtility.add(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}},
                new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualAddResult.length);
        assertArrayEquals(new double[]{20.0d, 1.0d, 20.0d, 1.0d}, actualAddResult[0], 0.0);
    }

    /**
     * Method under test: {@link MatrixUtility#subtract(double[][], double[][])}
     */
    @Test
    void testSubtract() {
        // Arrange and Act
        double[][] actualSubtractResult = MatrixUtility.subtract(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}},
                new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}});

        // Assert
        assertEquals(1, actualSubtractResult.length);
        assertArrayEquals(new double[]{0.0d, 0.0d, 0.0d, 0.0d}, actualSubtractResult[0], 0.0);
    }

    /**
     * Method under test: {@link MatrixUtility#multiply(double[], double)}
     */
    @Test
    void testMultiply() {
        // Arrange, Act and Assert
        assertArrayEquals(new double[]{100.0d, 5.0d, 100.0d, 5.0d},
                MatrixUtility.multiply(new double[]{10.0d, 0.5d, 10.0d, 0.5d}, 10.0d), 0.0);
    }

    /**
     * Method under test: {@link MatrixUtility#multiply(double[][], double)}
     */
    @Test
    void testMultiply2() {
        // Arrange and Act
        double[][] actualMultiplyResult = MatrixUtility.multiply(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}},
                10.0d);

        // Assert
        assertEquals(1, actualMultiplyResult.length);
        assertArrayEquals(new double[]{100.0d, 5.0d, 100.0d, 5.0d}, actualMultiplyResult[0], 0.0);
    }
}
