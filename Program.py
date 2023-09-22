# Importamos las librerias de Python necesarias
import numpy as np
from skimage.io import imshow, imread
import cv2
import os
import matplotlib.pyplot as plt


def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        cv2.imwrite(ruta_img_crop, image_crop)
        

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
    except Exception as e:
        print("Ha ocurrido un error:", str(e))


def mostrar_img(image, titulo):
    cv2.imshow(titulo,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Guardar la imagen recortada en la ruta indicada
        

def image_to_matrix(image):
    numpydata = np.asarray(image)
    return numpydata


    
#main

#parte1
imagen1 = cv2.imread("bradpitt.jpg")
imagen2 = cv2.imread("pradera.jpg")
mostrar_img(imagen1, "imagen brad pitt")
mostrar_img(imagen2, "imagen pradera")



#parte 2
print(" El tamaño de la imagen 1 es de" + str(imagen1.shape))
print(" El tamaño de la imagen 2 es de" + str(imagen2.shape))


#Parte 3
recortar_imagen_v2("pradera.jpg", "praderacortada.jpg", 0,400,0,400 )
imagen2_recor= cv2.imread("praderacortada.jpg")
mostrar_img(imagen2_recor, "imagen recortada pradera")

recortar_imagen_v2("bradpitt.jpg", "bradpittrebanado.jpg", 200,600,200,600 )
imagen1_recor= cv2.imread("bradpittrebanado.jpg")
mostrar_img(imagen1_recor, "imagen brad pitt recortado")


#Parte 4
matriz_imagen1 = image_to_matrix(imagen1_recor)
mostrar_img(matriz_imagen1, "kasndanf")
print(np.size(matriz_imagen1))
matriz_imagen2 = image_to_matrix(imagen2_recor)
print(np.size(matriz_imagen2))


#Parte 5
matriz_transpuesta1 = np.transpose(matriz_imagen1)
mostrar_img(matriz_transpuesta1, "ajdkjsalfj")
print(matriz_transpuesta1)

matriz_transpuesta2 = np.transpose(matriz_imagen2)
print(matriz_transpuesta2)


