import random

# Parâmetros do Problema
TAMANHO_POPULACAO = 10
TAMANHO_GENOMA = 8
TAXA_MUTACAO = 0.1
GERACOES = 20
CAPACIDADE_MOCHILA = 15

# Definição dos itens: cada item tem um peso e um valor (peso, valor)
itens = [(2, 3), (3, 4), (4, 5), (5, 8), (9, 10), (4, 7), (2, 6), (1, 2)]

# Índice do item com valor sentimental (item obrigatório)
INDICE_SENTIMENTAL = 4  # Item com peso 9 e valor 10
PENALIDADE_SENTIMENTAL = 5  # Redução no valor total

# Função de Aptidão (fitness)
def aptidao(individuo):
    peso_total = 0
    valor_total = 0
    
    for i in range(TAMANHO_GENOMA):
        if individuo[i] == 1:
            peso, valor = itens[i]
            peso_total += peso
            valor_total += valor

    # Aplica a penalidade se o peso exceder a capacidade
    if peso_total > CAPACIDADE_MOCHILA:
        return 0  # Soluções inviáveis têm aptidão zero

    # Verifica se o item sentimental está incluído
    if individuo[INDICE_SENTIMENTAL] == 0:
        return 0  # Soluções que não incluem o item sentimental são inválidas

    # Aplica a penalidade do item sentimental
    valor_total -= PENALIDADE_SENTIMENTAL

    return valor_total

# Criar um indivíduo (genoma)
def criar_individuo():
    while True:
        individuo = [random.randint(0, 1) for _ in range(TAMANHO_GENOMA)]
        # Garante que o item sentimental esteja sempre incluído
        individuo[INDICE_SENTIMENTAL] = 1
        if aptidao(individuo) > 0:
            return individuo

# Criar uma população inicial
def criar_populacao():
    return [criar_individuo() for _ in range(TAMANHO_POPULACAO)]

# Cruzamento (Crossover)
def cruzamento(pai1, pai2):
    ponto_corte = random.randint(1, TAMANHO_GENOMA - 1)
    filho = pai1[:ponto_corte] + pai2[ponto_corte:]
    # Garante que o item sentimental esteja sempre incluído
    filho[INDICE_SENTIMENTAL] = 1
    return filho

# Mutação
def mutacao(individuo):
    for i in range(TAMANHO_GENOMA):
        if i == INDICE_SENTIMENTAL:
            continue  # Não muta o item sentimental
        if random.random() < TAXA_MUTACAO:
            individuo[i] = 1 - individuo[i]
    return individuo

# Seleção
def selecao(populacao):
    populacao_ordenada = sorted(populacao, key=lambda ind: aptidao(ind), reverse=True)
    return populacao_ordenada[:TAMANHO_POPULACAO // 2]

# Evolução
def evoluir(populacao):
    nova_populacao = []
    pais_selecionados = selecao(populacao)
    while len(nova_populacao) < TAMANHO_POPULACAO:
        pai1 = random.choice(pais_selecionados)
        pai2 = random.choice(pais_selecionados)
        filho = cruzamento(pai1, pai2)
        filho = mutacao(filho)
        if aptidao(filho) > 0:
            nova_populacao.append(filho)
    return nova_populacao

# Algoritmo Genético
populacao = criar_populacao()
for geracao in range(GERACOES):
    populacao = evoluir(populacao)
    melhor_individuo = max(populacao, key=lambda ind: aptidao(ind))
    print(f"Geração {geracao + 1}: Melhor Aptidão = {aptidao(melhor_individuo)}")

# Melhor solução encontrada
print("Melhor solução: ", melhor_individuo)
print("Itens selecionados:")
for i in range(TAMANHO_GENOMA):
    if melhor_individuo[i] == 1:
        print(f"Item {i + 1}: Peso = {itens[i][0]}, Valor = {itens[i][1]}")
