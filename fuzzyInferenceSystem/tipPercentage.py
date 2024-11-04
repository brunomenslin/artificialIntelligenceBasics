import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Definição das variáveis fuzzy
meal = ctrl.Antecedent(np.arange(0, 11, 1), 'meal')  # Avaliação da refeição (0-10)
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')  # Avaliação do serviço (0-10)
service_time = ctrl.Antecedent(np.arange(0, 11, 1), 'service_time')  # Tempo de serviço (0-10)
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')  # Gorjeta (0-25%)

# Definição das funções de pertinência para cada variável
meal['bland'] = fuzz.trimf(meal.universe, [0, 0, 5])
meal['tasty'] = fuzz.trimf(meal.universe, [5, 10, 10])

service['bad'] = fuzz.trimf(service.universe, [0, 0, 5])
service['excellent'] = fuzz.trimf(service.universe, [5, 10, 10])

service_time['slow'] = fuzz.trimf(service_time.universe, [0, 0, 5])
service_time['average'] = fuzz.trimf(service_time.universe, [3, 5, 7])
service_time['fast'] = fuzz.trimf(service_time.universe, [5, 10, 10])

tip['no_tip'] = fuzz.trimf(tip.universe, [0, 0, 5])
tip['small'] = fuzz.trimf(tip.universe, [5, 10, 15])
tip['generous'] = fuzz.trimf(tip.universe, [15, 20, 25])

# Regras fuzzy
rule1 = ctrl.Rule(meal['bland'] & service['bad'], tip['small'])
rule2 = ctrl.Rule(meal['tasty'] & service['excellent'], tip['generous'])
rule3 = ctrl.Rule(service_time['slow'], tip['no_tip'])
rule4 = ctrl.Rule(service_time['average'] | service_time['fast'], tip['small'])

# Controle do sistema fuzzy
tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

# Entrada dos valores
meal_value = 7  # Avaliação da refeição
service_value = 8  # Avaliação do serviço
time_value = 4  # Tempo de serviço
tipping.input['meal'] = meal_value
tipping.input['service'] = service_value
tipping.input['service_time'] = time_value

# Cálculo da gorjeta
tipping.compute()
print(f"Gorjeta recomendada: {tipping.output['tip']:.2f}%")

# Gráficos em uma única figura
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Gráfico da avaliação da refeição
axs[0, 0].plot(meal.universe, meal['bland'].mf, 'b', linewidth=1.5, label='Bland')
axs[0, 0].plot(meal.universe, meal['tasty'].mf, 'g', linewidth=1.5, label='Tasty')
axs[0, 0].set_title('Avaliação da Refeição')
axs[0, 0].legend()

# Gráfico da avaliação do serviço
axs[0, 1].plot(service.universe, service['bad'].mf, 'r', linewidth=1.5, label='Bad')
axs[0, 1].plot(service.universe, service['excellent'].mf, 'g', linewidth=1.5, label='Excellent')
axs[0, 1].set_title('Avaliação do Serviço')
axs[0, 1].legend()

# Gráfico do tempo de serviço
axs[1, 0].plot(service_time.universe, service_time['slow'].mf, 'r', linewidth=1.5, label='Slow')
axs[1, 0].plot(service_time.universe, service_time['average'].mf, 'b', linewidth=1.5, label='Average')
axs[1, 0].plot(service_time.universe, service_time['fast'].mf, 'g', linewidth=1.5, label='Fast')
axs[1, 0].set_title('Tempo de Serviço')
axs[1, 0].legend()

# Gráfico da gorjeta
axs[1, 1].plot(tip.universe, tip['no_tip'].mf, 'r', linewidth=1.5, label='No Tip')
axs[1, 1].plot(tip.universe, tip['small'].mf, 'b', linewidth=1.5, label='Small Tip')
axs[1, 1].plot(tip.universe, tip['generous'].mf, 'g', linewidth=1.5, label='Generous Tip')
axs[1, 1].set_title('Gorjeta')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
