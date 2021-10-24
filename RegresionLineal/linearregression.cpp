#include "linearregression.h"
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

/* Se necesita entrenar el modelo, lo que implica minimizar alguna funcion de costo
 * y de esta forma se puede medir la precision de la funcion de hipotesis. La funcion
 * de costo es la forma de penalizar al modelo por cometer un error
 */

float LinearRegression::FuncionCosto(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta){

    Eigen::MatrixXd diferencia = pow((X * theta - y).array(), 2);
    return (diferencia.sum()/(2 * X.rows()));

}

/* Se implementa la funcion para dar al algoritmo los valores de theta iniciales, que
 * cambiaran iterativamente hasta que converga al valor minimo de la funcion de costo.
 * Basicamente describira el gradiente descendiente: Es dado por la derivada parcial
 * de la funcion. Esta tiene un alpha que representa el salto del gradiente y el numero
 * de iteraciones que se necesitan para actualizar theta hasta que la funcion converja
 * al minimo esperado
 */

std::tuple<Eigen::VectorXd, std::vector<float>> LinearRegression::GradienteDescendiente(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta, float alpha, int iteraciones){
    /*Almacenamiento temporal de valores de theta*/
    Eigen::MatrixXd temporal = theta;

    /*Variable con la cantidad de parametros m (features) */
    int parametros = theta.rows();

    /*Variable para ubicar el costo inicial, que se actualiza iterativamente con los pesos*/
    std::vector<float> costo;
    costo.push_back(FuncionCosto(X, y, theta));

    /*Por cada iteracion se calcula la funcion de error */

    for(int i=0;i<iteraciones;++i){
        Eigen::MatrixXd error = X*theta - y;
        for(int j=0;j<parametros;++j){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = theta(j,0) - ((alpha/X.rows())*termino.sum());
        }
        theta=temporal;
        costo.push_back(FuncionCosto(X, y, theta));
    }

    return std::make_tuple(theta, costo);
}
