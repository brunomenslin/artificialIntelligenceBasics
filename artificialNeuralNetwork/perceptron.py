# Perceptron!
# Forma simples de rede neural artificial, utilizada para classificação binária.
# Treinado para classificar quatro padrões de entrada, o algoritmo ajusta os
# pesos até que o erro entre a *saída real* e a *saída desejada* seja pequeno.

    # Objetivo!
    # Treinar um perceptron (rede neural com um único neurônio) para classificar
    # um conjunto de entradas em duas classes (1 ou -1).

    # Entradas!
    # São valores binários expandidos com um bias (sempre -1).

    # Saídas!
    # Classificação como 1 ou -1.

    # Processo!
    # A rede ajusta os pesos iterativamente com base no erro até que a diferença
    # entre as saídas desejadas e as calculadas seja mínima.


# Bibliotecas Necessárias!
import numpy


# Função de Ativação!
# Realiza a classificação das saídas em duas classes, A e B.
def activation(weightedSum):
    if weightedSum >= 0:
        return 1 # Classe A "verdadeiro".
    else:
        return -1 # Classe B "falso".


# Função de Saída!
# Calcula a saída, retornando a classificação das entradas e pesos que recebe.
def findOutput(inputs, weights):
    weightedSum = 0 # Valor escalar "u".

    # Calcula a soma ponderada das entradas, multiplicando cada entrada do vetor
    # "inputs" pelos pesos correspondentes no vetor "weights".
    # Os pesos são ajustados conforme a taxa de aprendizado.
    for i in range(0, len(inputs)):
        weightedSum += (weights[i] * inputs[i])

    # Retorna a classificação das entradas e pesos.
    return activation(weightedSum)


# Taxa de Aprendizado!
# Determina a magnitude do ajuste nos pesos a cada iteração.
# Um valor maior acelera o aprendizado, mas pode causar instabilidade.
# Um valor menor torna o aprendizado mais lento.
learningRate = 0.01


# Erro Desejado!
# Deve afetar o número de iterações necessárias para convergência do algoritmo.
# O treinamento termina quando o erro acumulado da rede for inferior ao desejado.
desiredErrors = 0.001


# Função de Treino!
# Treina o perceptron e ajusta os pesos com base no erro entre a saída atual do
# perceptron e a saída desejada, para as entradas e saídas esperadas.
def trainPerceptron (inputs, desiredOutputs, weights):

    # Número de Iterações!
    # A cada época, loop while, a rede neural é treinada com todos os inputs.
    iterations = 0

    # Laço de Treinamento!
    while True:

        # Erro Acumulado da Rede!
        errors = 0.0

        # Loop para as amostras de entrada.
        for i in range(0, len(inputs)):

            # Para cada amostra de entrada, calcula-se a saída atual.
            output = findOutput(inputs[i], weights)

            # O erro acumulado representa a diferença quadrática entre a saída
            # desejada "desiredOutputs" e a saída calculada "output".
            errors += ((desiredOutputs[i] - output) ** 2) / 2

            # Definição do sinal de aprendizado, que é a diferença entre a saída
            # desejada "desiredErrors" e a saída atual "output" multiplicada
            # pela taxa de aprendizado "learningRate".
            learningSignal = (learningRate * (desiredOutputs[i] - output))

            # Os pesos são ajustados de acordo com o sinal de aprendizado
            # "learningSignal" para aproximar a saída desejada.
            for k in range(0, len(inputs[i])):
                weights[k] += (learningSignal * inputs[i][k])

        # Cálculo de iterações.
        iterations += 1

        # Registro de treinamento.
        print(errors, weights)

        # O treinamento continua até que o erro acumulado "errors" seja inferior ao
        # erro desejado "desiredErrors", o que indica que a rede aprendeu
        # a classificar corretamente as entradas.
        if errors < desiredErrors:
            print('Iterations:', iterations)
            break

    # Retorna os pesos ajustados.
    return weights


# Função de Teste!
# Após o treinamento, o código testa o perceptron com as mesmas entradas usadas
# durante o treinamento para verificar se a rede agora classifica corretamente.
def testPerceptron (adjustedWeights, inputs):
    for i in range(len(inputs)):
        print(findOutput(inputs[i], adjustedWeights))


# Par de entradas binárias (x1 e x2) mais o bias, elemento extra fixo para -1.
# Bias é uma forma de ajustar a função de ativação para melhor separação linear.
# Estes conjuntos estão relacionados a problemas de lógica binária.
inputs = [[1, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, -1]]


# Saíddas Desejadas!
# As saídas desejadas para cada porta lógica (or, and, xor) representam
# o comportamento que esperamos que o perceptron aprenda,
# com cada porta tendo sua lógica específica.
desiredOutputsForOr = [1, 1, 1, -1]
desiredOutputsForAnd = [1, -1, -1, -1]
desiredOutputsForXor = [-1, 1, 1, -1]


# Pesos!
# Inicialização randômica dos pesos.
weights = numpy.random.rand(len(inputs[0]))

# Apenas um perceptron não consegue aprender a função XOR, visto que, esta não é
# linearmente separável, não há uma linha reta que possa dividir os pontos
# de entrada em duas classes distintas, como na função de ativação.
testPerceptron(trainPerceptron(inputs, desiredOutputsForOr, weights), inputs)
testPerceptron(trainPerceptron(inputs, desiredOutputsForAnd, weights), inputs)
# testPerceptron(trainPerceptron(inputs, desiredOutputsForXor, weights), inputs)
