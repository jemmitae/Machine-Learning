# E10 - Random Forest Performance Review

Read and comment the paper *Do we Need Hundreds of Classifiers to Solve Real World Classification Problems?*

### Reference:
http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf


La lectura corresponde a una evaluacion de un modelo, con el cual esperaban definir que cantidad de clasificadores eran necesarios para responder cuestionamientos del mundo real. Tomaron en cuenta un total de 179 clasificadores de 17 tipos de modelos (análisis discriminante, Bayesiano,redes neuronales, máquinas de vectores de soporte, árboles de decisión, clasificadores basados ​​en reglas, impulso,
ensacado, apilamiento, bosques aleatorios y otros conjuntos, modelos lineales generalizados, vecinos más cercanos, mínimos cuadrados parciales y regresión de componentes principales, regresión logística y multinomial, splines de regresión adaptativa múltiple y otros métodos), Usaron los siguientes lenguajes Weka, R (con y sin el paquete de caret), C y Matlab, incluidos todos los clasificadores disponibles hoy. Con 121 conjuntos de datos, que representan la totalidad de los datos de UCI.
Identificaron que el bosque aleatorio es claramente la mejor familia de clasificadores (3 de los 5 mejores clasificadores son RF),
seguido de SVM (4 clasificadores en el top-10), redes neuronales y conjuntos de refuerzo (5 y 3 miembros en el top-20, respectivamente).
En la mayoria de casos cuando se necesita la solucion de un problema de clasificacion, el analista se va por la opción que mejor conoce o que se le hace la que representara mejor los datos, es posible que un investigador no pueda usar clasificadores que surjan de áreas en el que no se sea un experto (por ejemplo, para desarrollar el ajuste de parámetros), siendo a menudo limitado al uso de los métodos dentro de su dominio de experiencia. Sin embargo, no hay certeza, que funcionan mejor, para un conjunto de datos dado.

Es por ello que la falta de implementación disponible para muchos clasificadores es un gran inconveniente, aunque se ha reducido parcialmente debido a la gran cantidad de clasificadores implementados. dentro de toda la coleccion de modelos, se concluye que la "Familia" de random Forest es la que mejor clasifica los datos, por encima de unas cuantas decimas por los otros modelos. finalmente como lenguaje concluyen que la mayoría de los mejores clasificadores están implementados en R y afinados, usando caret, que parece la mejor alternativa para seleccionar una implementación de clasificador.