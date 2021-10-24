#include "exeigennorm.h"
#include "linearregression.h"

#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>

/* En primer lugar se creara una clase llamada "ExEigenNorm", la cual nos permitira leer un dataset
 * extraer y montar sobre la estructura Eigen para normalizar los datos del dataset */

int main(int argc, char *argv[])
{
    /* Se crea un objeto del tipo ExEigenNorm, se incluyen los 3 argumentos del constructor:
     * nombre del dataset, delimitador y flag (si tiene o no cabecera) */

    ExEigenNorm extraccion(argv[1], argv[2], argv[3]);
    LinearRegression LR;


    /* Se leen los datos del archivo por la funcion LeerCSV() */

    std::vector<std::vector<std::string>> dataFrame = extraccion.LeerCSV();

    /* Para probar la segunda funcion CSVtoEigen se define la cantidad de filas y columnas basados
     * en los datos de entrada */

    int filas = dataFrame.size()+1;
    int columnas = dataFrame[0].size();

    Eigen::MatrixXd matrizDataFrame = extraccion.CSVtoEigen(dataFrame, filas, columnas);

    //std::cout<<matrizDataFrame<<std::endl;

    /* Para desarrollar el primer algoritmo de regresion lineal, donde se probara con los datos de los
     * vinos (winedata.csv) se presentara la regresion lineal para multiples variables. Dada la naturaleza de
     * la regresion lineal, si se tiene variables con diferentes unidades una variable puede modificar, estropear
     * otra variable. Para esto se estandarizan los datos dejando a todas las variables en el mismo
     * orden de manitud y centradas en 0. Para ello se construira una funcion de normalizacion basada en
     * el setscore normalizacion. Se necesitan 3 funciones: La funcion de normalizacion, la del promedio
     * y la de la desviacion estandar */

    /* Se muestran los datos normalizados */

    Eigen::MatrixXd normMatriz = extraccion.Normalizacion(matrizDataFrame);

    //std::cout << normMatriz << std::endl;

    /* Visualizacion del promedio
    auto miPromedio = extraccion.Promedio(matrizDataFrame);

    std::cout<<miPromedio<<std::endl<<std::endl; */

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> divDatos = extraccion.TrainTestSplit(normMatriz, 0.8);

    /* Se desempaca la tupla, se usa std::tie
     * https://en.cppreference.com/w/cpp/utility/tuple/tie */

    Eigen::MatrixXd X_Train, y_Train, X_Test, y_Test;

    std::tie(X_Train, y_Train, X_Test, y_Test) = divDatos;

    /* Inspeccion visual de la division de datos de entrenamiento y prueba */

    /* std::cout << "Tamanho original del dataset:     " << normMatriz.rows() << std::endl;
    std::cout << "Tamanho entrenamiento (filas):    " << X_Train.rows() << std::endl;
    std::cout << "Tamanho entrenamiento (columnas): " << X_Train.cols() << std::endl;
    std::cout << "Tamanho prueba (filas):           " << X_Test.rows() << std::endl;
    std::cout << "Tamanho prueba (columnas):        " << X_Test.cols() << std::endl;

    std::cout << std::endl;

    std::cout << "Tamanho original del dataset:     " << normMatriz.rows() << std::endl;
    std::cout << "Tamanho entrenamiento (filas):    " << y_Train.rows() << std::endl;
    std::cout << "Tamanho entrenamiento (columnas): " << y_Train.cols() << std::endl;
    std::cout << "Tamanho prueba (filas):           " << y_Test.rows() << std::endl;
    std::cout << "Tamanho prueba (columnas):        " << y_Test.cols() << std::endl; */

    /* A continuacion se procede a probar la clase de regresion lineal */

    Eigen::VectorXd vectorTrain = Eigen::VectorXd::Ones(X_Train.rows());
    Eigen::VectorXd vectorTest = Eigen::VectorXd::Ones(X_Test.rows());

    /* Redimension de las matrices para ubicacion en los vectores de Ones (reshape con NumPy) */

    X_Train.conservativeResize(X_Train.rows(), X_Train.cols() + 1);
    X_Train.col(X_Train.cols()-1) = vectorTrain;

    X_Test.conservativeResize(X_Test.rows(), X_Test.cols() + 1);
    X_Test.col(X_Test.cols()-1) = vectorTest;

    /* Se define el vector theta que pasara al algoritmo del gradiente descendiente un Vector de CEROS
     * del mismo tamaño del vector de entrenamiento. Adicional, se pasara alpha y el numero de iteraciones */

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_Train.cols());
    float alpha =0.01;
    int iteraciones = 1000;

    /* Se definen las variables de salida que representan los coeficientes y el vector de costo */

    Eigen::VectorXd thetaOut;
    std::vector<float> costo;

    std::tuple<Eigen::VectorXd, std::vector<float>> gradienteDesc = LR.GradienteDescendiente(X_Train, y_Train, theta, alpha, iteraciones);

    std::tie(thetaOut, costo) = gradienteDesc;

    /* Se imprimen los valores de los coeficientes theta para cada feature */

    //std::cout<<"\nTheta: \n\n"<<thetaOut<<"\n";
    /* std::cout<<"\nCosto: \n\n";
    for(auto valor:costo){
        std::cout<<valor<<"\n";
    } */

    /* Exportamos a fichero Costo y Theta */

    extraccion.VectorToFile(costo, "costo.txt");
    extraccion.EigenToFile(thetaOut, "theta.txt");


    return EXIT_SUCCESS;
}
