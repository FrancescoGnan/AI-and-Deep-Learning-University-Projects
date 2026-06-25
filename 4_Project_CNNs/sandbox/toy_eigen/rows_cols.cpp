//
// Created by Ekkehard Steinmacher on 15.03.25.
//

// Compile with: g++ -std=c++20 -o rows_cols.exe rows_cols.cpp -I../../eigen

#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define arbitrary sized matrix
    // first arg: #rows, second arg: #cols
    Eigen::MatrixXd mat(3, 3);
    mat << 1, 2, 3,
           4, 5, 6,
           7, 8, 9;

    // Output matrix
    std::cout << "Matrix:\n" << mat << "\n\n";

    // Output number of rows and columns
    std::cout << "Number of rows: " << mat.rows() << "\n";
    std::cout << "Number of columns: " << mat.cols() << "\n";

//    Eigen::MatrixXd exp_mat = mat.array().exp();
//    std::cout << "Matrix:\n" << exp_mat << std::endl;

//    Eigen::VectorXd sum_rowwise = mat.rowwise().sum();
//    std::cout << "Sum rowwise:\n" << sum_rowwise << std::endl;

//    Eigen::VectorXd sum_colwise = mat.colwise().sum();
//    std::cout << "Sum colwise:\n" << sum_colwise << std::endl;

//    Eigen::VectorXd max_colwise = mat.colwise().maxCoeff();
//    std::cout << "Max colwise:\n" << max_colwise << std::endl;

    // Subtract the maximum values from each element in the corresponding row
//    Eigen::MatrixXd shifted_tensor = mat.rowwise() - max_colwise.transpose();
//    std::cout << "Shifted tensor:\n" << shifted_tensor << std::endl;

//    Eigen::MatrixXd prediction_tensor = mat.array().rowwise() / sum_colwise.array().transpose();
//    std::cout << "Prediction tensor:\n" << prediction_tensor << std::endl;


    Eigen::MatrixXd mat2(3, 3);
    mat2 << 2, 2, 2,
            1, 1, 1,
            0, 1, 0;
//    Eigen::MatrixXd ele_mult = (mat2.array() * mat.array()); //.colwise().sum();
//    std::cout << "Return tensor:\n" << ele_mult << std::endl;

    Eigen::VectorXd vec_colsum = (mat2.array() * mat.array()).colwise().sum();
    std::cout << "Column sum:\n" << vec_colsum << std::endl;

    Eigen::MatrixXd mat_minus = mat2.rowwise() - vec_colsum.transpose();
    std::cout << "Matrix minus:\n" << mat_minus << std::endl;

    Eigen::MatrixXd return_tensor = mat.array() * mat_minus.array();
    std::cout << "Return tensor:\n" << return_tensor << std::endl;

    return 0;
}
