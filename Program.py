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
        

def image_to_matrix(image):
    numpydata = np.asarray(image)
    return numpydata

def es_invertible(matriz):
    return np.linalg.det(matriz) != 0


def calcular_inversa(matriz):
    if es_invertible(matriz):
        return np.linalg.inv(matriz)
    else:
        return None
    
#main

#parte1
imagen1 = cv2.imread("bradpitt.jpg")
imagen2 = cv2.imread("pradera.jpg")
mostrar_img(imagen1, "imagen Brad Pitt")
mostrar_img(imagen2, "imagen pradera")



#parte 2
print("Parte 2")
print(" El tamaño de la imagen 1 es de" + str(imagen1.shape))
print(" El tamaño de la imagen 2 es de" + str(imagen2.shape))
print("\n")


#Parte 3
print("Parte 3")
recortar_imagen_v2("pradera.jpg", "praderacortada.jpg", 0,250,0,250 )
imagen2_recor= cv2.imread("praderacortada.jpg")
mostrar_img(imagen2_recor, "imagen recortada pradera")

recortar_imagen_v2("bradpitt.jpg", "bradpittrebanado.jpg", 300,550,300,550 )
imagen1_recor= cv2.imread("bradpittrebanado.jpg")
mostrar_img(imagen1_recor, "imagen brad pitt recortado")
print("\n")


#Parte 4
print("Parte 4")
matriz_imagen1 = image_to_matrix(imagen1_recor)
print(matriz_imagen1)
tamano1 = np.size(matriz_imagen1)
print("El tamaño de la imagen 1 es: ", tamano1)
matriz_imagen2 = image_to_matrix(imagen2_recor)
print("\n")
tamano2 = np.size(matriz_imagen2)
print("El tamaño de la imagen 2 es: ", tamano2)
print("\n")


#Parte 5
imagen_traspuesta1 = imagen1_recor.transpose((1,0,2))
matriz_traspuesta1 = image_to_matrix(imagen_traspuesta1)
mostrar_img(imagen_traspuesta1, "imagen 1 traspuesta ")
print("Parte 5 imagen 1 ")
print(matriz_traspuesta1)
print("\n")

imagen_traspuesta2 = imagen2_recor.transpose((1,0,2))
matriz_traspuesta2 = image_to_matrix(imagen_traspuesta2)
mostrar_img(imagen_traspuesta2, "imagen 2 traspuesta")
print("Parte 5 imagen 2")
print(matriz_traspuesta2)
print("\n")


#Parte 6
grayscale1 = cv2.cvtColor(imagen1_recor, cv2.COLOR_BGR2GRAY)
mostrar_img(grayscale1, "imagen 1 grayscale")

grayscale2 = cv2.cvtColor(imagen2_recor, cv2.COLOR_BGR2GRAY)
mostrar_img(grayscale2, "imagen 2 grayscale")

#Parte 7
matriz_gray1 = image_to_matrix(grayscale1)
print("Parte 7")

if (es_invertible(matriz_gray1)):
    inversa1 = calcular_inversa(matriz_gray1)
    print("La inverda de la matriz 1 es:")
    print(inversa1)
    print("\n")


matriz_gray2 = image_to_matrix(grayscale2)

if (es_invertible(matriz_gray2)):
    inversa2 = calcular_inversa(matriz_gray2)
    print("La inversa de la matriz 2 es:")
    print(inversa2)
    print("\n")




#Parte 8
multi = matriz_gray1 * 3
multi = np.clip(multi,0, 255)
print("\n")
print("Parte 8")
print("La matriz de la imagen 1 multiplicada por un escalar 3:")
print(multi)
print("\n")


#Parte 9
tamano = matriz_gray1.shape

identidad = np.eye(tamano[0])
w = np.fliplr(identidad)

print("Parte 9")
multi = matriz_gray1 * w 
print("La matriz de la imagen 1 multiplicada por la matriz antidiagonal: ")
print(multi)
print("\n")
multi2 = w * matriz_gray1
print("La matriz de la imagen 2 multiplicada por la matriz antidiagonal: ")
print(multi2)
print("\n")

#parte 10

matriz_aux = np.full((tamano[0],tamano[1]),255,dtype=np.uint8)
print("Parte 10")
print("La matriz negativa de la imagen 1")
print(matriz_aux - matriz_gray1)
print("\n")
mostrar_img(matriz_aux - matriz_gray1,"Imagen 1 negativa")

print("La matriz negativa de la imagen 2")
print(matriz_aux - matriz_gray2)
print("\n")
mostrar_img(matriz_aux - matriz_gray1,"Imagen 2 negativa")



