#include "exeigennorm.h"

#include <vector>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

/* Primera funcion: lectura de ficheros csv
 * vector de vectores String
 * La idea es leer linea por linea y almacenar en un vector de vectores tipo String
 */

std::vector<std::vector<std::string>> ExEigenNorm::LeerCSV(){

    /* Se abre el archivo para lectura solamente */
    std::ifstream Archivo(setDatos);

    /* Vector de vectores de tipo string que tendra los datos del dataset */
    std::vector<std::vector<std::string>> datosString;

    /* Se itera a traves de cada linea del dataset, y se divide el contenido
     * dado por el delimitador provisto por el constructor */

    std::string linea="";

    while(getline(Archivo, linea)){
        std::vector<std::string> vectorFila;
        /* Dividimos el vectorFila por cada uno de los delimitadores */
        boost::algorithm::split(vectorFila, linea, boost::is_any_of(delimitador));
        datosString.push_back(vectorFila);
    }

    /* Se cierra el archivo (fichero) */
    Archivo.close();

    /* Se retorna el vector de vectores de tipo string */
    return datosString;

}

/* Se crea la segunda funcion para guardar el vector de vectores tipo String
 * a una matriz Eigen. Similar a Pandas de python para presentr el dataframe*/

Eigen::MatrixXd ExEigenNorm::CSVtoEigen(std::vector<std::vector<std::string>> datosString, int filas, int col){
    /* Si tiene cabecera (True) la removemos */
    if(header == true){
        filas-=1;
    }

    /* Se itera sobre filas y columnas para almacenar en la matriz vacia (tamaño = filas * columnas)
     * que basicamente almacenara Strings en un vector: luego lo pasaremos a float para ser manipulados
     */

    Eigen::MatrixXd dfMatriz(col, filas);

    for (int i=0; i<filas;i++){
        for (int j=0; j<col;j++){
            dfMatriz(j,i) = atof(datosString[i][j].c_str());
        }
    }

    return dfMatriz.transpose();
}

/* A continuacion se implementan las funciones necesarias para la normalizacion */

/* En C++ auto es como dynamic en dart - especifica que el tipo de la variable que se empieza a declarar
 * se deducira automaticamente de su inicializador y, para las funciones si su tipo de retorno es auto,
 * se evaluara mediante la expresion del tipo de retorno en tiempo de ejecucion */

/* auto ExEigenNorm::Promedio(Eigen::MatrixXd datos){
    //Se ingresa como entrada la matriz de datos (datos) y regresa el promedio
    return datos.colwise().mean();
}*/

/* Todavia no se sabe que retorna datos.colwise().mean(): En C++ la herencia del tipo de dato
 * no es directa o no se sabe que tipo de dato debe retornar, entonces para ello se declara el tipo
 * en una expresion 'decltype' con el fin de tener seguridad de que tipo de dato retornara la
 * funcion */

auto ExEigenNorm::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean()){
    /* Se ingresa como entrada la matriz de datos (datos) y regresa el promedio */
    return datos.colwise().mean();
}

/* Para implementar la funcion de desviacion estandar
 * datos = x_i   -    promedio(x) */

auto ExEigenNorm::Desviacion(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()){
    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}

/* Normalizacion Z-Score es una estrategia de normalizacion de datos que evita el problema de los outliers */

Eigen::MatrixXd ExEigenNorm::Normalizacion(Eigen::MatrixXd datos){

    Eigen::MatrixXd datos_escalados=datos.rowwise()-Promedio(datos);    //(x_i - promedio)


    Eigen::MatrixXd matrixNorm = datos_escalados.array().rowwise()/Desviacion(datos_escalados);


    return matrixNorm;
}

 /* A continuacion se hara una funcion para dividir los datos en conjunto de datos de entrenamiento y de prueba
  */

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> ExEigenNorm::TrainTestSplit(Eigen::MatrixXd datos, float trainSize){
    int filas = datos.rows();
    int filasTrain = round(trainSize*filas);
    int filasTest = filas-filasTrain;

    /* Con eigen se puede especificar un bloque de una matriz, por ejemplo se pueden seleccionar las filas superiores
     * para el conjunto de entrenamiento indicando cuantas filas se desean, se selecciona desde 0 (fila 0) hasta
     * el numero de filas indicado
     */

    Eigen::MatrixXd entrenamiento = datos.topRows(filasTrain);

    /* Seleccionadas las filas superiores para entrenamiento, se seleccionan las 11 primeras columnas que representan
     * las variables indepentiendes o features
     */

    Eigen::MatrixXd X_train = entrenamiento.leftCols(datos.cols()-1);

    /* Se selecciona ahora la variable dependiente que corresponde a la ultima columna
     */
    Eigen::MatrixXd y_train = entrenamiento.rightCols(1);

    /* Se hace lo mismo para el conjunto de pruebas */

    Eigen::MatrixXd prueba = datos.topRows(filasTest);
    Eigen::MatrixXd X_test = prueba.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_test = prueba.rightCols(1);

    /* Finalmente se retorna la tupla con el conjunto de datos de prueba y entrenamiento */

    return std::make_tuple(X_train, y_train, X_test, y_test);

}

/* Se implementan 2 funciones para exportar a ficheros desde vector y desde Eigen */

void ExEigenNorm::VectorToFile(std::vector<float> vector, std::string nombreArchivo){
    std::ofstream fichero(nombreArchivo);
    std::ostream_iterator<float> iterador(fichero, "\n");
    std::copy(vector.begin(), vector.end(), iterador);
}

void ExEigenNorm::EigenToFile(Eigen::VectorXd datos, std::string nombreArchivo){
    std::ofstream fichero(nombreArchivo);
    if(fichero.is_open()){
        fichero<<datos<<"\n";
    }
}

