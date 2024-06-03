package data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DataReaderDiffblueTest {

    /**
     * Method under test: {@link DataReader#readData(String)}
     */
    @Test
    void testReadData() {
        // Arrange and Act
        List<Image> actualReadDataResult = DataReader.readData("Path");

        // Assert
        assertTrue(actualReadDataResult.isEmpty());
    }
}
