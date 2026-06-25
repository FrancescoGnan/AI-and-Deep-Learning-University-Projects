CXX = g++
CXXFLAGS = -std=c++20 -I../../eigen
TARGET = test_eigen.exe
SRC = test_eigen.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)