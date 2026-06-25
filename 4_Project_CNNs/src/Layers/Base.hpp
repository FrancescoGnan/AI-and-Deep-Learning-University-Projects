// BaseLayer is the parent class of all other layers

/* Needed varaibles and functions:
* trainable parameter (bool)
* weights (Matrix (Eigen::MatrixXd))
* forward function (Eigen::MatrixXd)
* backward function (Eigen::MatrixXd)
* print_info function
 * */

/* Notes:
* The BaseLayer class does not need explicit constructors, since it is purely abstrct.
  Only the derived classes need constructors.
* */

#pragma once

#include <Eigen/Dense>
#include <memory>

// class BaseLayer
class BaseLayer
{
    public:
        // Constructor
        BaseLayer() : trainable(false), weights(std::make_unique<Eigen::MatrixXd>()) {}

        // Destructor
        virtual ~BaseLayer() {std::cout << "BaseLayer destroyed" << std::endl; } // virtual ~BaseLayer() = default;

        // Functions
        virtual std::unique_ptr<Eigen::MatrixXd> forward(const Eigen::MatrixXd& input) = 0; // = 0 indicates pure virtual funciton that has to be overwriten later

        virtual std::unique_ptr<Eigen::MatrixXd> backward(const Eigen::MatrixXd& error) = 0;

        virtual void print_info() = 0;

        // Member variables
        bool trainable;
        std::unique_ptr<Eigen::MatrixXd> weights;
};