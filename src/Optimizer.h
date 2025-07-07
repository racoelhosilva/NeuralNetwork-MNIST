#pragma once

#include "Matrix.h"
#include <memory>

/**
 * @namespace optimizer
 * @brief Contains classes and functions for various optimization algorithms.
 */
namespace optimizer {

    /**
     * @enum Type
     * @brief Enum representing different types of optimizers.
     */
    enum class Type {
        SGD,        ///< SGD Stochastic Gradient Descent optimizer
        Momentum,   ///< Momentum optimizer
        RMSProp,    ///< RMSProp optimizer
        Adam        ///< Adam optimizer
    };

    /**
     * @struct Settings
     * @brief Contains settings for the optimizer.
     */
    struct Settings {
        optimizer::Type type = Type::SGD;   ///< Type of optimizer to use (default = SGD)
        double epsilon = 1e-8;              ///< Small constant to avoid division by zero (default = 1e-8)
        double beta1 = 0.9;                 ///< Exponential decay rate for first moment estimates (default = 0.9)
        double beta2 = 0.999;               ///< Exponential decay rate for second moment estimates (default = 0.999)
    };

    /**
     * @class Base
     * @brief Base class for all optimizers.
     */
    class Base {
    public:
        /**
         * @brief Default destructor.
         */
        virtual ~Base() = default;

        /**
         * @brief Updates the parameters using the optimizer's algorithm.
         * @param m The matrix containing the parameters to be updated.
         * @param grad The gradient of the loss with respect to the parameters.
         * @param lr The learning rate for the update.
         */
        virtual void update(Matrix& m, const Matrix& grad, double lr) = 0;
    };

    /**
     * @class SGD
     * @brief Implements the Stochastic Gradient Descent optimization algorithm.
     */
    class SGD final : public Base {
    public:
        /**
         * @brief Updates the parameters using the SGD algorithm.
         * @param param The matrix containing the parameters to be updated.
         * @param grad The gradient of the loss with respect to the parameters.
         * @param lr The learning rate for the update.
         */
        void update(Matrix& param, const Matrix& grad, double lr) override;
    };

    /**
     * @class Momentum
     * @brief Implements the Momentum optimization algorithm.
     */
    class Momentum final : public Base {
    public:
        /**
         * @brief Constructs a Momentum optimizer with the specified dimensions and momentum factor.
         * @param rows Number of rows in the parameter matrix.
         * @param cols Number of columns in the parameter matrix.
         * @param momentum Momentum factor.
         */
        Momentum(int rows, int cols, double momentum)
            : velocity(rows, cols), momentum {momentum} {}

        /**
         * @brief Updates the parameters using the Momentum algorithm.
         * @param param The matrix containing the parameters to be updated.
         * @param grad The gradient of the loss with respect to the parameters.
         * @param lr The learning rate for the update.
         */
        void update(Matrix& param, const Matrix& grad, double lr) override;
    private:
        /**
         * @brief The velocity matrix used in the Momentum algorithm.
         */
        Matrix velocity;

        /**
         * @brief The momentum factor used in the Momentum algorithm.
         */
        double momentum;
    };

    /**
     * @class RMSProp
     * @brief Implements the RMSProp optimization algorithm.
     */
    class RMSProp final : public Base {
    public:
        /**
         * @brief Constructs an RMSProp optimizer with the specified dimensions, decay factor, and epsilon.
         * @param rows Number of rows in the parameter matrix.
         * @param cols Number of columns in the parameter matrix.
         * @param decay Decay factor for the moving average of squared gradients.
         * @param epsilon Small constant to avoid division by zero.
         */
        RMSProp(int rows, int cols, double decay, double epsilon)
            : cache(rows, cols), decay {decay}, epsilon {epsilon} {}
        
        /**
         * @brief Updates the parameters using the RMSProp algorithm.
         * @param param The matrix containing the parameters to be updated.
         * @param grad The gradient of the loss with respect to the parameters.
         * @param lr The learning rate for the update.  
         */
        void update(Matrix& param, const Matrix& grad, double lr) override;
    private:
        /**
         * @brief The cache matrix used in the RMSProp algorithm to store the moving average of squared gradients.
         */
        Matrix cache;
        
        /**
         * @brief The decay factor for the moving average of squared gradients.
         */ 
        double decay;
        
        /**
         * @brief Small constant to avoid division by zero.
         */
        double epsilon;
    };

    /**
     * @class Adam
     * @brief Implements the Adam optimization algorithm.
     */
    class Adam final : public Base {
    public:
        /**
         * @brief Constructs an Adam optimizer with the specified dimensions, beta1, beta2, and epsilon.
         * @param rows Number of rows in the parameter matrix.
         * @param cols Number of columns in the parameter matrix.
         * @param beta1 Exponential decay rate for the first moment estimates.
         * @param beta2 Exponential decay rate for the second moment estimates.
         * @param epsilon Small constant to avoid division by zero.
         */
        Adam(int rows, int cols, double beta1, double beta2, double epsilon)
            : m(rows, cols), v(rows, cols)
            , beta1 {beta1}, beta2 {beta2}, epsilon {epsilon}
        {}

        /**
         * @brief Updates the parameters using the Adam algorithm.
         * @param param The matrix containing the parameters to be updated.
         * @param grad The gradient of the loss with respect to the parameters.
         * @param lr The learning rate for the update.
         */
        void update(Matrix& param, const Matrix& grad, double lr) override;
    private:
        /**
         * @brief The first moment estimates matrix used in the Adam algorithm.
         */
        Matrix m;

        /**
         * @brief The second moment estimates matrix used in the Adam algorithm.
         */
        Matrix v;
        
        /**
         * @brief Exponential decay rate for the first moment estimates.
         */
        double beta1;

        /**
         * @brief Exponential decay rate for the second moment estimates.
         */
        double beta2;
        
        /**
         * @brief Small constant to avoid division by zero.
         */
        double epsilon;

        /**
         * @brief Cache for the first moment estimates.
         */
        double cache_p1 = 1.0;

        /**
         * @brief Cache for the second moment estimates.
         */
        double cache_p2 = 1.0;

        /**
         * @brief Time step counter.
         */
        int t = 0;
    };

    /**
     * @brief Creates an optimizer based on the specified settings.
     * @param rows Number of rows in the parameter matrix.
     * @param cols Number of columns in the parameter matrix.
     * @param optimizer Settings for the optimizer to create.
     * @return A shared pointer to the created optimizer.
     * @throws std::invalid_argument if the optimizer type is unknown.
     */
    [[nodiscard]] std::shared_ptr<optimizer::Base> create(
        int rows, int cols, 
        const optimizer::Settings& optimizer
    );
}
