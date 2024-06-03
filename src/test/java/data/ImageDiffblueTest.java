package data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;

class ImageDiffblueTest {
    /**
     * Methods under test:
     * <ul>
     *   <li>{@link Image#Image(double[][], int)}
     *   <li>{@link Image#getData()}
     *   <li>{@link Image#getLabel()}
     * </ul>
     */
    @Test
    void testGettersAndSetters() {
        // Arrange
        double[][] data = new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}};

        // Act
        Image actualImage = new Image(data, 1);
        double[][] actualData = actualImage.getData();

        // Assert
        assertEquals(1, actualImage.getLabel());
        assertSame(data, actualData);
    }

    /**
     * Method under test: {@link Image#toString()}
     */
    @Test
    void testToString() {
        // Arrange, Act and Assert
        assertEquals("1\n10.0,0.5,10.0,0.5,\n",
                (new Image(new double[][]{new double[]{10.0d, 0.5d, 10.0d, 0.5d}}, 1)).toString());
    }
}
