//
// Created by Ekkehard Steinmacher on 20.03.25.
//

// Compile with: g++ -std=c++20 -o stod.exe stod.cpp

#include <iostream>
#include <string>

int main() {
    std::string str = "1E-3";
    double value = std::stod(str);

    std::cout << "Converted value: " << value << std::endl; // Output: 0.001

    return 0;
}