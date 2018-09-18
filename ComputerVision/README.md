V2.0:  
Resumo:  
Identificação do objeto utilizando SURF (Speeded-Up Robust Features)+Brute Force Matches  
Rastreamento do objeto utilizando rastreador CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)  
Fusão do identificador e do rastreador utilizando Filtro de Kalman Extendido (EKF)    


Algoritmo:  
-Inicia matrizes do EKF  
Inicia matrizes de confiança do modelo Q e da medição R  
Matriz de relação entre medição e estados H=Identidade, portanto não aparece nas equações 
   
-Inicialização  
Usuário: define objeto (botões 1 e 2) e seleciona região do objeto  
Inicializa estado (X=(x1,y1,x2,y2)'), com seleção do usuário e matriz de covariâncias P  
Inicia rastreador com região selecionada  
Obtém pontos chaves e descritores da região selecionada  

-Etapa de predição do EKF   
Predição da região nova (X_k+1|k) obtida pelo rastreador CSRT  
Atualização da matriz de covariâncias P = F*P*F'+Q com F = k*diagonal(dx1,dy1,dx2,dy2)+I, sendo dn a variação da variável de estado n   

-Etapa de atualização do EKF  
SURF+Brute Force Matcher obtém correspondência entre pontos da imagem e pontos do template inicial  
Não havendo correspondências suficientes segue pela predição  
Havendo correspondências suficientes carrega matriz de medida e obtém matriz de diferença entre medição e predição  
y = (x1,y1,x2,y2) medidos, q=y-X_k+1|k  
Calcula-se a distância de Mahalanobis para a medição (q*inv(P+R)*q')  
Se a distância estiver abaixo de 9,488 (confiabilidade de 95% para 4 graus de liberdade na tabela chi quadrado) a medição é aceita  
do contrário segue apenas pela predição  
Se a medição for aceita atualiza matriz de estado e de covariâncias  
K = P*inv(P+R)  
X_k+1|k+1 = X_k+1|k+K*q  
P = (I-K)*P*(I-K)'+K*R*K'  
Reajusta rastreador para a nova região  
