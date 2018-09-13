V1.0:  
Rastreamento de um objeto definido pelo usuário utilizando SURF e Brute Force Matcher  
Clica e arrasta sobre a imagem permite ao usuário selecionar template do objeto a ser rastreado  
Algoritmo SURF (Speeded-Up Robust Features) obtém descritores do template salvo e do frame mais recente  
Algoritmo Brute Force Matcher obtém correspondência entre descritores do template e do frame  
Homografia transformação entre orientação e dimensão do objeto no frame e no template  
Aplica-se transformada obtida ao retângulo limite do template e desenha o retângulo transformado sobre a imagem  