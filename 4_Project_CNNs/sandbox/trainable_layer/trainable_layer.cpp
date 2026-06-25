//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -o trainable_layer.exe trainable_layer.cpp

#include <iostream>
#include <random>

class Trainable {
private:
    double trainable_variable;

    // Helper function to generate a random double between 0 and 1
    static double get_random_value() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dis(0.0, 1.0);
        return dis(gen);
    }

public:
    // Default constructor
    Trainable() : trainable_variable(get_random_value()) {
        std::cout << "Trainable object created with initial value: " << trainable_variable << std::endl;
    }

    // Copy constructor
    Trainable(const Trainable& other) : trainable_variable(other.trainable_variable) {
        std::cout << "Trainable object copied with value: " << trainable_variable << std::endl;
    }

    // Move constructor
    Trainable(Trainable&& other) noexcept : trainable_variable(other.trainable_variable) {
        std::cout << "Trainable object moved with value: " << trainable_variable << std::endl;
    }

    // Destructor
    ~Trainable() {
        std::cout << "Trainable object with value " << trainable_variable << " is being destroyed." << std::endl;
    }

    // Modify function to set trainable_variable to 42
    void modify(double value) {
        trainable_variable = value;
        std::cout << "Trainable variable modified to: " << trainable_variable << std::endl;
    }

    // Getter for trainable_variable
    double get_value() const {
        return trainable_variable;
    }
};

// Pass by reference to avoid copying the object, thus the object is not destroyed
void change_value(Trainable& obj, double value) {
    obj.modify(value);
}

int main() {
    Trainable obj;
    obj.get_value();
    change_value(obj, 42.0);
    obj.get_value();
    change_value(obj, 100.0);
    obj.get_value();
    return 0;
}