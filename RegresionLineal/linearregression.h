#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

class LinearRegression
{
public:
    LinearRegression(){}

    float FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones);
};

#endif // LINEARREGRESSION_H
