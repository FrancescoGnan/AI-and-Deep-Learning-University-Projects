#pragma once

#include <Eigen/Dense>

Eigen::MatrixXd addRowOfOnes(const Eigen::MatrixXd& mat) {
    // Get the current number of rows and columns
    int rows = mat.rows();
    int cols = mat.cols();

    // Create a new matrix with an extra row
    Eigen::MatrixXd extendedMat(rows + 1, cols);

    // Copy the original matrix
    extendedMat.block(0, 0, rows, cols) = mat;

    // Set the last row to ones
    extendedMat.row(rows) = Eigen::VectorXd::Ones(cols);

    return extendedMat;
}

Eigen::MatrixXd removeLastRow(const Eigen::MatrixXd& mat) {
    // Ensure the matrix has at least one row to remove
    if (mat.rows() <= 1) {
        throw std::invalid_argument("Matrix must have more than one row to remove the last row.");
    }

    // Create a new matrix without the last row
    return mat.topRows(mat.rows() - 1);
}