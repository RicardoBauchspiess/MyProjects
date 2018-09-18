V2.0:  
Resumo:  
Identifica��o do objeto utilizando SURF (Speeded-Up Robust Features)+Brute Force Matches  
Rastreamento do objeto utilizando rastreador CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)  
Fus�o do identificador e do rastreador utilizando Filtro de Kalman Extendido (EKF)    


Algoritmo:  
-Inicia matrizes do EKF  
Inicia matrizes de confian�a do modelo Q e da medi��o R  
Matriz de rela��o entre medi��o e estados H=Identidade, portanto n�o aparece nas equa��es 
   
-Inicializa��o  
Usu�rio: define objeto (bot�es 1 e 2) e seleciona regi�o do objeto  
Inicializa estado (X=(x1,y1,x2,y2)'), com sele��o do usu�rio e matriz de covari�ncias P  
Inicia rastreador com regi�o selecionada  
Obt�m pontos chaves e descritores da regi�o selecionada  

-Etapa de predi��o do EKF   
Predi��o da regi�o nova (X_k+1|k) obtida pelo rastreador CSRT  
Atualiza��o da matriz de covari�ncias P = F*P*F'+Q com F = k*diagonal(dx1,dy1,dx2,dy2)+I, sendo dn a varia��o da vari�vel de estado n   

-Etapa de atualiza��o do EKF  
SURF+Brute Force Matcher obt�m correspond�ncia entre pontos da imagem e pontos do template inicial  
N�o havendo correspond�ncias suficientes segue pela predi��o  
Havendo correspond�ncias suficientes carrega matriz de medida e obt�m matriz de diferen�a entre medi��o e predi��o  
y = (x1,y1,x2,y2) medidos, q=y-X_k+1|k  
Calcula-se a dist�ncia de Mahalanobis para a medi��o (q*inv(P+R)*q')  
Se a dist�ncia estiver abaixo de 9,488 (confiabilidade de 95% para 4 graus de liberdade na tabela chi quadrado) a medi��o � aceita  
do contr�rio segue apenas pela predi��o  
Se a medi��o for aceita atualiza matriz de estado e de covari�ncias  
K = P*inv(P+R)  
X_k+1|k+1 = X_k+1|k+K*q  
P = (I-K)*P*(I-K)'+K*R*K'  
Reajusta rastreador para a nova regi�o  
