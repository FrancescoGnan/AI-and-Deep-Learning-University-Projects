# See https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/

CXX=g++
CFLAGS= -std=c++20 -O3 -g -Ieigen

SRC_DIR=src
OBJ_DIR=obj
EXEC_DIR=exes
SRC_FILES=$(SRC_DIR)/IO/read_dataset_images.cpp $(SRC_DIR)/IO/io.cpp $(SRC_DIR)/IO/read_dataset_labels.cpp $./NeuralNetwork.cpp
OBJ_FILES=$(OBJ_DIR)/IO/read_dataset_images.o $(OBJ_DIR)/IO/io.o $(OBJ_DIR)/IO/read_dataset_labels.o $(OBJ_DIR)/NeuralNetwork.o

all: images_io labels_io NN

images_io: $(OBJ_DIR)/IO/read_dataset_images.o $(OBJ_DIR)/IO/io.o
	$(CXX) -o $(EXEC_DIR)/$@ $(OBJ_DIR)/IO/read_dataset_images.o $(OBJ_DIR)/IO/io.o

labels_io: $(OBJ_DIR)/IO/read_dataset_labels.o $(OBJ_DIR)/IO/io.o
	$(CXX) -o $(EXEC_DIR)/$@ $(OBJ_DIR)/IO/read_dataset_labels.o $(OBJ_DIR)/IO/io.o

NN: $(./)NeuralNetwork.cpp $(SRC_DIR)/IO/io.cpp
	$(CXX) $(CFLAGS) -o $(EXEC_DIR)/NeuralNetwork.exe $(./)NeuralNetwork.cpp $(SRC_DIR)/IO/io.cpp

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)/IO # ensures that obj folder exists -> if not: gets created
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/IO/*.o
	rm -rf $(EXEC_DIR)/*