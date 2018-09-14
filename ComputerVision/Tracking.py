# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:56:40 2018

@author: Ricardo
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt



#função de retorno do mouse para obter templates, grava os pontos inicial e final da região desejada
point = []
cropping = False
done = False
def click_and_crop(event, x, y, flags, param):
    global cropping
    global point, done
 
    #inicia uma região com o aberto do botão esquerdo do mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [(x, y), (x, y)]
        cropping = True
        first_crop = False
    #Termina a região quando o botão esquerdo do mouse for solto
    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        point[1] = (x, y)
        done = True
    #Atualiza a região final quando o mouse move permitindo o usuario visualizar a região que será selecionada    
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        point[1] = (x, y)
        


template_num = 0;
template1 = False
template2 = False

pred1 = [[0,0],[0,0]]
pred2 = [[0,0],[0,0]]

font = cv2.FONT_HERSHEY_SIMPLEX

#captura da webcam
video_capture = cv2.VideoCapture(0)

#grava video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("output.avi", fourcc, 15, (640, 480))

newframe = True
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", click_and_crop)

# Inicia detectores SIFT ou SURF

#sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(100)

#Inicia matcher brute force
MIN_MATCH_COUNT = 8
bf = cv2.BFMatcher()


while True:
    #pega novo frame da camera
    if newframe:
        _, frame = video_capture.read() # We get the last frame.
    temp_image = frame.copy()

    #Desenha retangulo na região selecionada pelo usuário
    if cropping == True and len(point) == 2:
        cv2.rectangle(temp_image, point[0], point[1], (0, 255, 0), 2)
    if template1 == True or template2 == True:
        #converte imagem para tons de cinza e obtém os pontos chaves e descritores do último frame
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = surf.detectAndCompute(gray_image,None)
        #kp2, des2 = sift.detectAndCompute(gray_image,None)
        if template1 == True:
            matches = bf.knnMatch(des1, des, k=2)
             #Salva as boas correspondências
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
                    
            if len(good)>MIN_MATCH_COUNT:
                #obtém pontos equivalentes entre template e frame
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
                #obtém transformação entre pontos do template e do frame
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if mask is not None:
                    matchesMask = mask.ravel().tolist()
                
                    #transforma retângulo do template em região equivalente no frame e desenha quadrilátero
                    h,w = gray_roi1.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    temp_image = cv2.polylines(temp_image,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)
         
                    cv2.putText(temp_image,"1",(dst[0,0,0],dst[0,0,1]), font, 0.8,(255,0,0),2,cv2.LINE_AA)
        if template2 == True:
            matches = bf.knnMatch(des2, des, k=2)
             #Salva as boas correspondências
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
                    
            if len(good)>MIN_MATCH_COUNT:
                #obtém pontos equivalentes entre template e frame
                src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
                #obtém transformação entre pontos do template e do frame
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if mask is not None:
                    matchesMask = mask.ravel().tolist()
                
                    #transforma retângulo do template em região equivalente no frame e desenha quadrilátero
                    h,w = gray_roi2.shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    temp_image = cv2.polylines(temp_image,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
                    cv2.putText(temp_image,'2',(dst[0,0,0],dst[0,0,1]), font, 0.8,(0,0,255),2,cv2.LINE_AA)

    if done == True and (point[0][1] != point[1][1] or point[0][0] != point[1][0]):
        #Se tiver a região selecionada pelo mouse salva template
        done = False
        if point[0][1] < point[1][1]:
            if point[0][0]<point[1][0]:
                roi = frame[point[0][1]:point[1][1]+1, point[0][0]:point[1][0]+1]
            else:
                roi = frame[point[0][1]:point[1][1]+1, point[1][0]:point[0][0]+1]
        else:
            if point[0][0]<point[1][0]:
                roi = frame[point[1][1]:point[0][1]+1, point[0][0]:point[1][0]+1]
            else:
                roi = frame[point[1][1]:point[0][1]+1, point[1][0]:point[0][0]+1]
        if template_num == 1:      
            cv2.imshow('ROI1', roi) #mostra template atual
            gray_roi1 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #converte imagem para tons de cinza
            #Encontra pontos chave e descritores do template com SIFT ou SURF
            kp1, des1 = surf.detectAndCompute(gray_roi1,None)
            #kp1, des1 = sift.detectAndCompute(gray_roi,None)
            template1 = True;
        elif template_num == 2:
            cv2.imshow('ROI2', roi) #mostra template atual
            gray_roi2 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #converte imagem para tons de cinza
            #Encontra pontos chave e descritores do template com SIFT ou SURF
            kp2, des2 = surf.detectAndCompute(gray_roi2,None)
            #kp1, des1 = sift.detectAndCompute(gray_roi,None)
            template2 = True;
        
    #mostra imagem na tela e salva em video
    cv2.putText(temp_image,str(template_num),(0,20), font, 0.8,(0,255,0),2,cv2.LINE_AA)
    video_writer.write(temp_image)
    cv2.imshow('Video', temp_image)
    
    
    #comandos do teclado
    c = cv2.waitKey(1)
    if c == 27: #botão ESC termina execução
        break  
    elif c == 115:  #botão 's' termina execução
        newframe =  not newframe
    elif c == ord("r") and  not cropping: #botão 'r' deleta template
        if template_num == 1:
            template1 = False
            cv2.destroyWindow('ROI1')
        elif template_num == 2:
            template2 = False
            cv2.destroyWindow('ROI2')
    elif c == ord("1"):
        template_num = 1
    elif c == ord("2"):
        template_num = 2
    elif c == ord("0"):
        template_num = 0

video_writer.release()
video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.