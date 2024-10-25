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


# conjunto de valores de entrada ampliados com a entrada dummy.
# A entrada é aumentada com um elemento extra fixo para -1 (bias).
# Bias é uma forma de ajustar a função de ativação para melhor separação linear.
inputs = [
    [1, 1, -1], # Par de entradas binárias (x1 e x2) mais o bias.
    [1, -1, -1], # Conjuntos relacionados a problemas de lógica binária.
    [-1, 1, -1],
    [-1, -1, -1]
]


# saidas desejadas
desiredOutputs = [1, 1, 1, -1]


# Pesos!
# Inicialização randômica dos pesos.
weights = numpy.random.rand(len(inputs[0]))


# Taxa de Aprendizado!
# Determina a magnitude do ajuste nos pesos a cada iteração.
# Um valor maior acelera o aprendizado, mas pode causar instabilidade.
# Um valor menor torna o aprendizado mais lento.
learningRate = 0.01


# Erro Desejado!
# Deve afetar o número de iterações necessárias para convergência do algoritmo.
# O treinamento termina quando o erro acumulado da rede for inferior ao desejado.
desiredErrors = 0.001


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


# Após o treinamento, o código testa o perceptron com as mesmas entradas usadas
# durante o treinamento para verificar se a rede agora classifica corretamente.
print(findOutput([1, 1, -1], weights))
print(findOutput([1, -1, -1], weights))
print(findOutput([-1, 1, -1], weights))
print(findOutput([-1, -1, -1], weights))
