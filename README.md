<p align="center">
<FONT FACE="times new roman" SIZE=5>
<i><b>Fundamentos de Machine Learning en Computación Avanzada</b></i>
<br>
<img src="https://res-5.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_256,w_256,f_auto,q_auto:eco/v1455514364/pim02bzqvgz0hibsra41.png"
width="150" height="150">
</img>
<br>
<i><b>Estudiantes:</b></i>
<br>
Santiago Nohrá Nieto
<br>
Juan Camilo Rodriguez Fonseca
<br>
<i><b>Docente:</b></i><br> John Jairo Corredor Franco
<br>
<i><b>Tema:</b></i><br> Parcial ML Wine
<br>
<i><b>Fecha inicio:</b></i><br> 20/10/21
<br>
<i><b>Fecha entrega:</b></i><br> 24/10/21
</FONT>
</p>

## Introducción
La regresión lineal es un proceso en el cual se estima un valor desconocido basándose en valores conocidos, bajo la condición de que este valor desconocido se encuentre usando únicamente multiplicación escalar y adición; es precisamente esto, lo que la hace lineal. En Machine Learning, este es uno de los algoritmos supervisados más populares, ya que es un método estadístico usado para análisis predictivo. La regresión lineal hace predicciones para variables continuas o numéricas, comúnmente relacionadas con ventas, salarios, precios, edades, y muchas otras más. 

El algoritmo de regresión lineal muestra una relación lineal entre una variable dependiente, usualmente denominada 'y', y una o más variables independientes, usualmente denominadas 'X'. Una vez realizado, este algoritmo de Machine Learning es en teoría capaz de predecir valores nuevos a partir del 'aprendizaje' realizado al momento de ser entrenado con datos ya existentes, es decir, este algoritmo puede predecir cómo el valor de la variable dependiente 'y' cambia a partir de los cambios en las variables independientes 'X'. En todos los casos, este modelo brinda una línea recta con pendiente que mejor se ajusta a la relación de los datos pertinentes. Sin embargo, esta recta es obtenida a partir del cálculo, por parte del modelo, de los coeficientes de la recta, que son los valores de la pendiente 'm' y el corte en el eje y 'b'. Hay muchas metodologías y pasos que permiten llegar a estos valores, sin embargo, para este proyecto, se realizó con el método del gradiente descendiente.

El gradiente descendiente en Machine Learning es un método en el cual se encuentra, a través de iteraciones, el mínimo de una función, que, comúnmente, es la función de costo generada a partir de los datos a ser tratados. 

La función de costo representa el error entre los valores predecidos por el modelo en la fase de entrenamiento y los valores reales. Esta función está dada por:
![image](https://user-images.githubusercontent.com/89982255/138610195-6a34aec1-b38e-410e-ba20-5cc706774e0d.png)

A partir de esto, el gradiente descendiente busca minimizar esta función y encontrar qué valores son los que cuentan con el menor grado de error en comparación con los valores originales y reales, a través del uso de derivadas parciales. Para mayor información se puede remitir a los archivos de este repositorio, los cuales están ampliamente comentados y explicados.

## Metodología
Para desarrollar este proyecto, resultó indispensable el uso de C++ y su compilador, Google Colaboratory, QT Creator, CMake y la biblioteca Eigen de C++. 

Adicional a esto, se usó el dataset wine.csv, que contiene atributos de un vino y su resultante calidad. Incluye la acidez fija, volátil, cantidad de ácido cítrico, azúcar residual, cloruros, dióxido de azufre, total de dióxido de azufre, densidad, pH, sulfatos, alcohol y por último, la calidad calificada de 1 a 8. En total, este dataset cuenta con 1599 valores en cada atributo.

Los ficheros de C++ fueron creados, editados y ejecutados usando QT creator. 

Primeramente, se desarrollaron una serie de métodos para el procesamiento de datos externos a ser usados en este proyecto, encontrados en el fichero '_exeigennorm.cpp_'. Esto incluye métodos capaces de leer un archivo CSV, pasarlo a una matriz de Eigen, calcular el promedio de los valores por columnas, calcular la desviación estándar por columnas, normalizar los datos del dataset, y separar los datos del conjuntos en conjuntos de entrenamiento y de prueba, y exportar a ficheros desde Eigen y Vector.

Seguido de esto, se implementaron una serie de funciones en el fichero '_linearregression.cpp_' capaces de calcular la función de costo y realizar el gradiente descendiente con los argumentos necesarios.

Por último, en el fichero '_main.cpp_', se implementaron todas estas clases y sus funciones con su respectivo header para poder procesar los datos del dataset mencionado previamente y llevar a cabo la regresión lineal usando las variables independientes, ubicadas en las primeras 11 columnas, y la variable independiente ubicada en la última columna. En este fichero se llevó a cabo la lectura del CSV, la conversión a una matriz de Eigen, la normalización del dataset, la división en conjuntos de entrenamiento y prueba, y el posterior cálculo del gradiente descendiente con valores iniciales y una tasa de aprendizaje determinada. Por último, estos resultados de costo y de theta, fueron exportados a unos ficheros .txt conteniendo todos sus valores.

Por otra parte, para verificar los coeficientes de la regresión usando SciKitLearn de python, se hizo un proceso similar al anterior. Primeramente los datos fueron analizados y procesados para llevar a cabo la regresión lineal, y posteriormente se calcularon los coeficientes de la regresión usando el mismo método utilizado en C++ (Gradiente descendiente) sin bibliotecas de python y luego usando SKLearn. Por último se compararon estos valores y se tomaron medidas de rendimiento, con las cuales se obtuvieron conclusiones sustanciales.

## Resultados
Como resultado de este proyecto se obtuvieron todos los archivos presentes en este repositorio, los cuales pueden ser libremente descargados, revisados y ejecutados.

## Conclusión
Con un análisis superficial de los resultados obtenidos, se puede decir que la regresión realizada en el modelo desarrollado en QT es levemente más precisa que la realizada con la biblioteca sklearn. Lo anterior dado que tiene mejores resultados en  2  de las  3  métricas obtenidas, específicamente en R2 Score y en Mean squared Error. (Véase el fichero .ipynb para resultados)

Adicionalmente, en la gráfica de la función de costo, se puede ver como se cumple con el objetivo, el cual es que el error disminuya a medida que se avanza en iteraciones. En la gráfica se puede ver que inicia aproximadamente en  0.5  y cuando terminan las iteraciones, está aproximadamente en 0.325.

## Notas adicionales
Para descargar y ejecutar el proyecto /RegresionLineal en QT Creator, es necesaria la definición de los kits a ser usados al momento de abrir el proyecto, para esto, se debe escoger sólo la opción de Debug en la categoría Build y darle la dirección /path/a/el/proyecto/RegresionLineal/Debug/, adicionalmente, en la parte de Run, se debe añadir en argumentos de línea de comandos el siguiente texto: '../winedata.csv "," header=True'.

Asimismo, para información más a fondo de todo lo que se hizo en el proyecto, se recomienda abrir los ficheros .cpp, .h y .ipynb, los cuales se encuentran ampliamente comentados y explicados en su totalidad.
