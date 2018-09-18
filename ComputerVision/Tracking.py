# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:56:40 2018

@author: Ricardo
"""
import os

import numpy as np
import cv2


os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" #Garante que webcam desliga por completo

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
        

#parametros de controle
template1 = False
template2 = False
newframe = True
template_num = 0

#captura da webcam
video_capture = cv2.VideoCapture(0)

#Inicia gravador de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter("output.avi", fourcc, 15, (640, 480))

#Inicia seletor de ROI
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", click_and_crop)

font = cv2.FONT_HERSHEY_SIMPLEX

#---------------Parâmetros Detector de Features
#Inicia SURF
surf = cv2.xfeatures2d.SURF_create(450)

#Inicia matcher brute force
MIN_MATCH_COUNT = 15
bf = cv2.BFMatcher()



#----------Parâmetros para filtro de Kalman
# Matriz de estados X limites da região rastreada: [xmin, ymin, xmax, ymax]'
# Matriz X inicializada pela primeira ROI selecionada pelo usuário
# f é função não linear de predição do alforitmo de rastremento
# X_k+1|k obtém valores diretamento do alforitmo de rastremento
# F será matriz diagonal com a variação de cada variável de estado
# P = Matriz de covariâncias, baixo valor inicial pois a região inicial é entrada do usuário
# Q = matriz de confiabilidade do modelo
Q = np.matrix( ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)) )
Q2 = np.matrix( ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)) )
# y = medição realizada pelo algoritmo de detecção por feature matching
# matriz H da medição é matriz identidade
# R = matriz de confiabilidade da medição, valor mais elevado devido às imprecisões do algoritmo de matching
R = np.matrix( ((100,0,0,0),(0,100,0,0),(0,0,10,0),(0,0,0,100)) )
R2 = np.matrix( ((100,0,0,0),(0,100,0,0),(0,0,10,0),(0,0,0,100)) )



while True:
    #pega novo frame da camera se não estiver em pausa
    if newframe:
        _, frame = video_capture.read() 
    temp_image = frame.copy()

    W,H = frame.shape[:2]
    #Desenha retangulo na região selecionada pelo usuário
    if cropping == True and len(point) == 2:
        cv2.rectangle(temp_image, point[0], point[1], (0, 255, 0), 2)
    if template1 == True or template2 == True:
        #converte imagem para tons de cinza e obtém os pontos chaves e descritores do último frame
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = surf.detectAndCompute(gray_image,None)
                   
        if template1 == True:
            deteccao = False
            #Etapa de predição do EKF
            (success, box) = tracker1.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                #cv2.rectangle(temp_image, (x, y), (x + w, y + h),(255, 255,0), 2)
                X_ant = X[:,:]            
                X = np.matrix(((x),(y),(x+w),(y+h))).transpose()            
                dx1 = np.abs(X[0,0]-X_ant[0,0])/10+1
                dy1 = np.abs(X[1,0]-X_ant[1,0])/10+1
                dx2 = np.abs(X[2,0]-X_ant[2,0])/10+1
                dy2 = np.abs(X[3,0]-X_ant[3,0])/10+1
                F = np.matrix( (( dx1,0,0,0),(0,dy1,0,0),(0,0,dx2,0),(0,0,0,dy2)) )                
                P = F*P*F.transpose()+Q
            
            #Etapa de Atualização do EKF
            
            #Obtém matches entre pontos chave
            matches = bf.knnMatch(des1, des, k=2)
            #Salva as boas correspondências
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            """
            #Desenha matches
            kt = []
            for m in good:
                kt.append(kp[m.trainIdx])
            temp_image = cv2.drawKeypoints(temp_image, kt,None,(255,0,0),4)
            """
            if len(good)>MIN_MATCH_COUNT:
                #obtém pontos equivalentes entre template e frame
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                #obtém transformação entre pontos do template e do frame
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if mask is not None and M is not None:
                    matchesMask = mask.ravel().tolist()
                
                    #transforma retângulo do template em região equivalente no frame
                    pts = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    #temp_image = cv2.polylines(temp_image,[np.int32(dst)],True,(255,0,0),3, cv2.LINE_AA)
                    
                    #obtém retângulo médio entre os pontos obtidos como medição do algoritmo
                    x_min = max(np.int32( (dst[0,0,0]+dst[1,0,0])/2 ),0)
                    y_min = max(np.int32( (dst[0,0,1]+dst[3,0,1])/2  ),0)
                    x_max = np.int32( (dst[2,0,0]+dst[3,0,0])/2 )
                    y_max = np.int32( (dst[1,0,1]+dst[2,0,1])/2 )
                    #if x_min != x_max and y_min != y_max:
                        #cv2.rectangle(temp_image, (x_min,y_min), (x_max,y_max), (255,0,0), 2)
                    
                    #obtém medição e diferença entre medição e predição
                    y = np.matrix(((x_min),(y_min),(x_max),(y_max))).transpose()                
                    q = y - X
                    #calcula distancia de Mahalanobis
                    Minv = np.linalg.inv(R+P)      
                    Mdist = q.transpose()*Minv*q
                    #Se a distancia de Mahalanobis estiver abaixo do valor de confiabilidade 95%
                    #para 4 graus de liberdade, a medição é aceita
                    if Mdist < 9.488:
                        deteccao = True
                        #Atualiza estado e matriz de covariâncias
                        K = P*Minv
                        X = X+K*q
                        P = (np.eye(4)-K)*P*(np.eye(4)-K).transpose()+K*R*K.transpose()
                        #Ajusta rastreador para nova posição
                        if X[0]<X[2] and X[1]<X[3]:
                            initBB = (X[0],X[1], X[2]+1-X[0],X[3]+1-X[1])
                            tracker1 = cv2.TrackerCSRT_create()
                            tracker1.init(frame, initBB)
            #verifica se o objeto saiu do frame e ignora rastreador em caso positivo
            if deteccao == False and (X[0]==0 or X[1] == 0 or X[2] >= W-1 or X[3] >= H-1):
                non_detect1 = non_detect1+1
            else:
                non_detect1 = 0
            if non_detect1<5:
                #Desenha retangulo na região definida
                cv2.putText(temp_image,"1",(np.int(X[0,0]),np.int(X[1,0])), font, 0.8,(255,0,0),2,cv2.LINE_AA)      
                cv2.rectangle(temp_image, (np.int(X[0,0]),np.int(X[1,0])), (np.int(X[2,0]),np.int(X[3,0])), (255,0,0), 2)
                                                                                                        

        if template2 == True:
            deteccao = False
            #Etapa de predição do EKF
            (success, box) = tracker2.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                #cv2.rectangle(temp_image, (x, y), (x + w, y + h),(0,255, 255), 2)
                X_ant = X2[:,:]            
                X2 = np.matrix(((x),(y),(x+w),(y+h))).transpose()            
                dx1 = np.abs(X2[0,0]-X_ant[0,0])/10+1
                dy1 = np.abs(X2[1,0]-X_ant[1,0])/10+1
                dx2 = np.abs(X2[2,0]-X_ant[2,0])/10+1
                dy2 = np.abs(X2[3,0]-X_ant[3,0])/10+1
                F = np.matrix( (( dx1,0,0,0),(0,dy1,0,0),(0,0,dx2,0),(0,0,0,dy2)) )                
                P2 = F*P2*F.transpose()+Q2
            
            #Etapa de Atualização do EKF
            
            #Obtém matches entre pontos chave           
            matches = bf.knnMatch(des2, des, k=2)
            #Salva as boas correspondências
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            """
            #Desenha matches
            kt = []
            for m in good:
                kt.append(kp[m.trainIdx])
            temp_image = cv2.drawKeypoints(temp_image, kt,None,(0,0,255),4)
            """
            if len(good)>MIN_MATCH_COUNT:
                #obtém pontos equivalentes entre template e frame
                src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
                #obtém transformação entre pontos do template e do frame
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                if mask is not None and M is not None:
                    matchesMask = mask.ravel().tolist()
                
                    #transforma retângulo do template em região equivalente no frame
                    pts = np.float32([ [0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    #temp_image = cv2.polylines(temp_image,[np.int32(dst)],True,(0,0,255),3, cv2.LINE_AA)
                                      
                    #obtém retângulo médio entre os pontos obtidos como medição do algoritmo
                    x_min = np.int32( (dst[0,0,0]+dst[1,0,0])/2 )
                    y_min = np.int32( (dst[0,0,1]+dst[3,0,1])/2  )
                    x_max = np.int32( (dst[2,0,0]+dst[3,0,0])/2 )
                    y_max = np.int32( (dst[1,0,1]+dst[2,0,1])/2 )
                    #if x_min != x_max and y_min != y_max:
                        #cv2.rectangle(temp_image, (x_min,y_min), (x_max,y_max), (0,0,255), 2)
                        
                    #obtém medição e diferença entre medição e predição
                    y = np.matrix(((x_min),(y_min),(x_max),(y_max))).transpose()                
                    q = y - X2
                    #calcula distancia de Mahalanobis
                    Minv = np.linalg.inv(R2+P2)      
                    Mdist = q.transpose()*Minv*q
                    #Se a distancia de Mahalanobis estiver abaixo do valor de confiabilidade 95%
                    #para 4 graus de liberdade, a medição é aceita
                    if Mdist < 9.488:
                        #Atualiza estado e matriz de covariâncias
                        deteccao = True
                        K = P2*Minv
                        X2 = X2+K*q
                        P2 = (np.eye(4)-K)*P2*(np.eye(4)-K).transpose()+K*R2*K.transpose()
                        #Ajusta rastreador para nova posição
                        if X2[0]<X2[2] and X2[1]<X2[3]:
                            initBB = (X2[0],X2[1], X2[2]+1-X2[0],X2[3]+1-X2[1])
                            tracker2 = cv2.TrackerCSRT_create()
                            tracker2.init(frame, initBB)
            #verifica se o objeto saiu do frame e ignora rastreador em caso positivo
            if deteccao == False and (X2[0]==0 or X2[1] == 0 or X2[2] >= W-1 or X2[3] >= H-1):
                non_detect2 = non_detect2+1
            else:
                non_detect2 = 0
            if non_detect2<5:
                #Desenha retangulo na região definida
                cv2.putText(temp_image,'2',(np.int(X2[0,0]),np.int(X2[1,0])), font, 0.8,(0,0,255),2,cv2.LINE_AA)
                cv2.rectangle(temp_image, (np.int(X2[0,0]),np.int(X2[1,0])), (np.int(X2[2,0]),np.int(X2[3,0])), (0,0,255), 2)
    if done == True and (point[0][1] != point[1][1] or point[0][0] != point[1][0]):
        #Se tiver a região selecionada pelo mouse salva template
        done = False
        if point[0][1] < point[1][1]:
            if point[0][0]<point[1][0]:
                roi = frame[point[0][1]:point[1][1]+1, point[0][0]:point[1][0]+1]

                initBB = (point[0][0],point[0][1],point[1][0]+1-point[0][0],point[1][1]+1-point[0][1])
            else:
                roi = frame[point[0][1]:point[1][1]+1, point[1][0]:point[0][0]+1]
                initBB = (point[0][0],point[1][1],point[1][0]+1-point[0][0],point[0][1]+1-point[1][1])
        else:
            if point[0][0]<point[1][0]:
                roi = frame[point[1][1]:point[0][1]+1, point[0][0]:point[1][0]+1]
                initBB = (point[1][0],point[0][1],point[0][0]+1-point[1][0],point[1][1]+1-point[0][1])
            else:
                roi = frame[point[1][1]:point[0][1]+1, point[1][0]:point[0][0]+1]
                initBB = (point[1][0],point[1][1], point[0][0]+1-point[1][0],point[0][1]+1-point[1][1])
                
        if template_num == 1:
            non_detect1 = 0
            template1 = True;
            #cv2.imshow('ROI1', roi) #mostra template atual
            gray_roi1 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #converte imagem para tons de cinza
            #Encontra pontos chave e descritores do template com SIFT ou SURF
            kp1, des1 = surf.detectAndCompute(gray_roi1,None)
            h1,w1 = gray_roi1.shape
            
            #Inicializa rastreador
            tracker1 = cv2.TrackerCSRT_create()
            tracker1.init(frame, initBB)
            
            #inicialização do filtro de Kalman estado X e matriz de covariancia P
            X = np.matrix(((initBB[0]),(initBB[1]),(initBB[2]+initBB[0]-1),(initBB[3]+initBB[1]-1))).transpose()
            P = np.matrix( ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)) )

            
        elif template_num == 2:
            non_detect2 = 0
            template2 = True;
            #cv2.imshow('ROI2', roi) #mostra template atual
            gray_roi2 = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) #converte imagem para tons de cinza
            #Encontra pontos chave e descritores do template com SIFT ou SURF
            kp2, des2 = surf.detectAndCompute(gray_roi2,None)
            h2,w2 = gray_roi2.shape
            
            #inicializa rastreador
            tracker2 = cv2.TrackerCSRT_create()
            tracker2.init(frame, initBB)
            
            #inicialização do filtro de Kalman estado X e matriz de covariancia P
            X2 = np.matrix(((initBB[0]),(initBB[1]),(initBB[2]+initBB[0]-1),(initBB[3]+initBB[1]-1))).transpose()
            P2 = np.matrix( ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)) )
            

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