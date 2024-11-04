# Agoritmos Genéticos
# Maximizar a soma de uma lista de bits (genes).
# Obtivo: é encontrar a combinação de bits (0 ou 1) que tenha a maior soma possível.
# Meta: Encontrar a sequência que maxima um função de aptidão.

import random

#Parâmetros
TAMANHO_POPULACAO = 10
TAMANHO_GENOMA = 8
GERACAES = 20
TAXA_MUTACAO = 0.05

# Função de Aptidão (fitness)
def aptidao(individuo):
  return sum(individuo)

# Criar um indivíduo (genoma)
def criar_individuo():
  return [random.randint(0, 1) for _ in range(TAMANHO_GENOMA)]

# Criar um população
def criar_populacao():
  return [criar_individuo() for _ in range(TAMANHO_POPULACAO)]

# Cruzamento
def cruzamento(pai1, pai2):
  ponto_corte = random.randint(1, TAMANHO_GENOMA-1)
  filho = pai1[:ponto_corte] + pai2[ponto_corte:]
  return filho

# Mutação
def mutacao(individuo):
  for i in range(TAMANHO_GENOMA):
    if random.random() < TAXA_MUTACAO:
      individuo[i] = 1 - individuo[i]
  return individuo

# Seleção
def selecao(populacao):
  populacao_ordenada=sorted(populacao, key=lambda ind: aptidao(ind), reverse=True)
  return populacao_ordenada[:TAMANHO_POPULACAO//2]

# Evolução
def evoluir(populacao):
  nova_populacao = []
  pais_selecionados = selecao(populacao)
  while len(nova_populacao) < TAMANHO_POPULACAO:
    pai1 = random.choice(pais_selecionados)
    pai2 = random.choice(pais_selecionados)
    filho = cruzamento(pai1, pai2)
    filho = mutacao(filho)
    nova_populacao.append(filho)
  return nova_populacao

# GA
populacao = criar_populacao()
for geracao in range(GERACAES):
  populacao = evoluir(populacao)
  melhor_individuo = max(populacao, key=lambda ind: aptidao(ind))
  print(f"Geração {geracao+1}: Melhor Aptidão = {aptidao(melhor_individuo)}")

# Melhor solução
print("Melhor solução: ", melhor_individuo)
