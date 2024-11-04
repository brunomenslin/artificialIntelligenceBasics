import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import random
import multiprocessing

# Definir semente aleatória para reprodutibilidade
np.random.seed(42)
random.seed(42)

def create_chromosome():
    """
    Cria um cromossomo codificando os parâmetros das funções de pertinência do controlador fuzzy.
    """
    # Vamos parametrizar as funções de pertinência para todas as variáveis de entrada.
    # Para cada variável de entrada, temos funções de pertinência definidas por formas trapezoidais ou triangulares.

    # Para simplificar, assumiremos funções de pertinência trapezoidais com parâmetros [a, b, c, d].
    # Vamos codificar os parâmetros para todas as funções de pertinência de todas as variáveis de entrada.

    # Variáveis de entrada e suas funções de pertinência:
    # - Ângulo do Pêndulo (angle): N, Z, P
    # - Velocidade Angular do Pêndulo (angular_velocity): N, Z, P
    # - Posição do Carrinho (position): N, Z, P
    # - Velocidade Linear do Carrinho (velocity): N, Z, P

    # Total de funções de pertinência: 4 variáveis * 3 funções cada = 12 funções
    # Cada função tem 4 parâmetros: total de genes = 12 * 4 = 48

    chromosome = []

    # Definir intervalos para cada variável
    variable_ranges = {
        'angle': [-0.2, 0.2],
        'angular_velocity': [-0.5, 0.5],
        'position': [-2, 2],
        'velocity': [-3, 3],
    }

    for var_name, var_range in variable_ranges.items():
        a_min, a_max = var_range

        # Para cada uma das três funções de pertinência (N, Z, P)
        for _ in range(3):
            # Gerar quatro parâmetros dentro do intervalo da variável
            params = np.random.uniform(a_min, a_max, 4)
            # Garantir que os parâmetros estejam ordenados corretamente (a <= b <= c <= d)
            params.sort()
            chromosome.extend(params)

    return np.array(chromosome)

def decode_chromosome(chromosome):
    """
    Decodifica o cromossomo em parâmetros de funções de pertinência para cada variável.
    """
    variable_params = {}
    index = 0

    variable_names = ['angle', 'angular_velocity', 'position', 'velocity']
    for var_name in variable_names:
        params = []
        for _ in range(3):  # Três funções de pertinência por variável
            mf_params = chromosome[index:index+4]
            params.append(mf_params)
            index += 4
        variable_params[var_name] = params

    return variable_params

def create_population(pop_size):
    """
    Cria uma população inicial de cromossomos.
    """
    population = [create_chromosome() for _ in range(pop_size)]
    return population

def evaluate_fitness(chromosome):
    """
    Avalia a aptidão de um cromossomo simulando o pêndulo.
    """
    variable_params = decode_chromosome(chromosome)

    # Executar simulação e calcular métricas de desempenho
    fitness = simulate_pendulum(variable_params)

    return fitness

def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Seleciona um indivíduo da população usando seleção por torneio.
    """
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected_fitness = [fitness_scores[i] for i in selected_indices]

    # Retorna o indivíduo com a melhor aptidão (menor aptidão é melhor)
    winner_index = selected_indices[np.argmin(selected_fitness)]
    return population[winner_index]

def crossover(parent1, parent2, crossover_rate):
    """
    Realiza crossover uniforme entre dois pais.
    """
    if random.random() < crossover_rate:
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()

def mutate(chromosome, mutation_rate):
    """
    Muta um cromossomo adicionando pequenos valores aleatórios.
    """
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            # Muta gene
            chromosome[i] += np.random.normal(0, 0.1)
            # Mantém gene dentro do intervalo da variável
            # Determina a qual variável este gene corresponde
            var_index = i // (4 * 3)  # 4 parâmetros por MF, 3 MFs por variável
            variable_names = ['angle', 'angular_velocity', 'position', 'velocity']
            var_name = variable_names[var_index]
            var_range = {
                'angle': [-0.2, 0.2],
                'angular_velocity': [-0.5, 0.5],
                'position': [-2, 2],
                'velocity': [-3, 3],
            }[var_name]
            chromosome[i] = np.clip(chromosome[i], var_range[0], var_range[1])
    return chromosome

def genetic_algorithm():
    """
    Executa o algoritmo genético para otimizar o controlador fuzzy.
    """
    # Parâmetros do GA
    pop_size = 20
    num_generations = 15
    crossover_rate = 0.8
    mutation_rate = 0.05
    elite_size = 2  # Número de indivíduos elite a serem mantidos

    # Cria população inicial
    population = create_population(pop_size)

    # Avalia aptidão inicial
    with multiprocessing.Pool() as pool:
        fitness_scores = pool.map(evaluate_fitness, population)

    best_fitness_over_time = []
    best_chromosome = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        print(f"Geração {generation+1}/{num_generations}")

        # Elitismo: Mantém os melhores indivíduos
        sorted_indices = np.argsort(fitness_scores)
        elite_individuals = [population[i] for i in sorted_indices[:elite_size]]

        new_population = elite_individuals.copy()

        # Gera novos indivíduos
        while len(new_population) < pop_size:
            # Seleção
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # Mutação
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

        # Avalia aptidão
        with multiprocessing.Pool() as pool:
            fitness_scores = pool.map(evaluate_fitness, population)

        # Registra melhor aptidão
        generation_best_fitness = min(fitness_scores)
        generation_best_index = np.argmin(fitness_scores)
        generation_best_chromosome = population[generation_best_index]

        print(f"Melhor Aptidão: {generation_best_fitness:.4f}")

        best_fitness_over_time.append(generation_best_fitness)

        if generation_best_fitness < best_fitness:
            best_fitness = generation_best_fitness
            best_chromosome = generation_best_chromosome.copy()

    # Plota aptidão ao longo das gerações
    plt.figure()
    plt.plot(range(1, num_generations+1), best_fitness_over_time, marker='o')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Aptidão')
    plt.title('Aptidão ao Longo das Gerações')
    plt.grid(True)
    plt.show()

    # Simula com os melhores parâmetros
    variable_params = decode_chromosome(best_chromosome)
    simulate_pendulum(variable_params, plot_results=True)

def simulate_pendulum(variable_params, plot_results=False):
    """
    Simula o pêndulo invertido usando os controladores fuzzy.
    Retorna a pontuação de aptidão.
    """
    # Define funções de pertinência
    inclination, angular_velocity, position, velocity, force = define_membership_functions(variable_params)

    # Define regras
    pendulum_rules = define_pendulum_rules(inclination, angular_velocity, force)
    cart_rules = define_cart_rules(position, velocity, force)

    # Cria o sistema de controle
    total_rules = pendulum_rules + cart_rules
    pendulum_ctrl = ctrl.ControlSystem(total_rules)
    pendulum_sim = ctrl.ControlSystemSimulation(pendulum_ctrl)

    # Parâmetros de simulação
    dt = 0.01  # Passo de tempo (segundos)
    t_max = 10  # Tempo total de simulação (segundos)
    num_steps = int(t_max / dt)  # Número de passos de simulação
    time = np.linspace(0, t_max, num_steps)  # Array de tempo

    # Parâmetros físicos do sistema de pêndulo invertido
    params = {
        'g': 9.81,   # Gravidade (m/s^2)
        'l': 1.0,    # Comprimento do pêndulo (m)
        'm_p': 0.1,  # Massa do pêndulo (kg)
        'm_c': 1.0   # Massa do carrinho (kg)
    }
    params['total_mass'] = params['m_p'] + params['m_c']

    # Inicializa variáveis de estado
    state = np.array([0.05, 0.0, 0.0, 0.0])  # [theta, theta_dot, x, x_dot]
    states = np.zeros((num_steps, 4))
    forces = np.zeros(num_steps)

    # Loop de simulação
    for i in range(num_steps):
        theta, theta_dot, x, x_dot = state
        states[i] = state

        # Define entradas para o FIS
        pendulum_sim.input['inclination'] = np.clip(theta, -0.2, 0.2)
        pendulum_sim.input['angular_velocity'] = np.clip(theta_dot, -0.5, 0.5)
        pendulum_sim.input['position'] = np.clip(x, -2, 2)
        pendulum_sim.input['velocity'] = np.clip(x_dot, -3, 3)

        # Calcula a força de saída
        try:
            pendulum_sim.compute()
            force_output = pendulum_sim.output['force']
        except:
            force_output = 0.0  # Aplica força zero se o cálculo falhar

        # Limita a força de saída
        force_output = np.clip(force_output, -300, 300)
        forces[i] = force_output

        # Atualiza o sistema
        state = update_system(state, force_output, dt, params)

        # Verifica se o pêndulo caiu
        if abs(theta) > np.pi / 2:
            # Penaliza aptidão por queda
            fitness = 1e6 + i  # Grande penalidade mais o tempo de falha
            if plot_results:
                print("Pêndulo caiu no tempo {:.2f}s".format(i * dt))
            break
    else:
        # Calcula métricas de desempenho
        fitness = performance_metrics(states, forces)

    if plot_results:
        plot_simulation_results(states, forces, time)

    return fitness

def performance_metrics(states, forces):
    """
    Calcula métricas de desempenho e retorna uma pontuação de aptidão.
    """
    dt = 0.01  # Passo de tempo
    theta = states[:, 0]
    x = states[:, 2]

    # Converte theta para graus
    theta_deg = np.rad2deg(theta)

    # Calcula RMSE do desvio do ângulo
    rmse_angle = np.sqrt(np.mean(theta_deg**2))

    # Esforço total de controle
    total_effort = np.sum(np.abs(forces)) * dt

    # Função de aptidão: combinação de RMSE e esforço de controle
    fitness = rmse_angle + 0.1 * total_effort  # Ajustar peso conforme necessário

    return fitness

def define_membership_functions(variable_params):
    """
    Define as funções de pertinência para todas as variáveis de entrada e saída usando os parâmetros do cromossomo.
    """
    # Desempacota os parâmetros
    inclination_params = variable_params['angle']
    angular_velocity_params = variable_params['angular_velocity']
    position_params = variable_params['position']
    velocity_params = variable_params['velocity']

    # A variável de força é mantida igual à implementação básica
    force = ctrl.Consequent(np.linspace(-300, 300, 601), 'force', defuzzify_method='centroid')
    force['NL'] = fuzz.trimf(force.universe, [-300, -300, -150])
    force['NM'] = fuzz.trimf(force.universe, [-150, -75, 0])
    force['Z'] = fuzz.trimf(force.universe, [-1, 0, 1])
    force['PM'] = fuzz.trimf(force.universe, [0, 75, 150])
    force['PL'] = fuzz.trimf(force.universe, [150, 300, 300])

    # Define Antecedentes
    inclination = ctrl.Antecedent(np.linspace(-0.2, 0.2, 401), 'inclination')
    angular_velocity = ctrl.Antecedent(np.linspace(-0.5, 0.5, 101), 'angular_velocity')
    position = ctrl.Antecedent(np.linspace(-2, 2, 401), 'position')
    velocity = ctrl.Antecedent(np.linspace(-3, 3, 601), 'velocity')

    # Define funções de pertinência para inclinação
    inclination['N'] = fuzz.trapmf(inclination.universe, inclination_params[0])
    inclination['Z'] = fuzz.trapmf(inclination.universe, inclination_params[1])
    inclination['P'] = fuzz.trapmf(inclination.universe, inclination_params[2])

    # Define funções de pertinência para velocidade angular
    angular_velocity['N'] = fuzz.trapmf(angular_velocity.universe, angular_velocity_params[0])
    angular_velocity['Z'] = fuzz.trapmf(angular_velocity.universe, angular_velocity_params[1])
    angular_velocity['P'] = fuzz.trapmf(angular_velocity.universe, angular_velocity_params[2])

    # Define funções de pertinência para posição
    position['N'] = fuzz.trapmf(position.universe, position_params[0])
    position['Z'] = fuzz.trapmf(position.universe, position_params[1])
    position['P'] = fuzz.trapmf(position.universe, position_params[2])

    # Define funções de pertinência para velocidade
    velocity['N'] = fuzz.trapmf(velocity.universe, velocity_params[0])
    velocity['Z'] = fuzz.trapmf(velocity.universe, velocity_params[1])
    velocity['P'] = fuzz.trapmf(velocity.universe, velocity_params[2])

    return inclination, angular_velocity, position, velocity, force

def define_pendulum_rules(inclination, angular_velocity, force):
    """
    Define as regras fuzzy para o controlador do pêndulo.
    """
    rules = []

    # Usa as mesmas regras da implementação básica
    # Matriz de regras para ângulo do pêndulo e velocidade angular
    # Linha N
    rules.append(ctrl.Rule(inclination['N'] & angular_velocity['N'], force['PL']))  # Empurrar forte para a esquerda
    rules.append(ctrl.Rule(inclination['N'] & angular_velocity['Z'], force['PM']))  # Empurrar para a esquerda
    rules.append(ctrl.Rule(inclination['N'] & angular_velocity['P'], force['Z']))   # Não empurrar

    # Linha Z
    rules.append(ctrl.Rule(inclination['Z'] & angular_velocity['N'], force['PM']))  # Empurrar para a esquerda
    rules.append(ctrl.Rule(inclination['Z'] & angular_velocity['Z'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(inclination['Z'] & angular_velocity['P'], force['NM']))  # Empurrar para a direita

    # Linha P
    rules.append(ctrl.Rule(inclination['P'] & angular_velocity['N'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(inclination['P'] & angular_velocity['Z'], force['NM']))  # Empurrar para a direita
    rules.append(ctrl.Rule(inclination['P'] & angular_velocity['P'], force['NL']))  # Empurrar forte para a direita

    return rules

def define_cart_rules(position, velocity, force):
    """
    Define regras fuzzy para o controlador do carrinho.
    """
    rules = []

    # Usa as mesmas regras da implementação básica
    # Matriz de regras para posição do carrinho e velocidade linear
    # Linha N
    rules.append(ctrl.Rule(position['N'] & velocity['N'], force['PL']))  # Empurrar forte para a direita
    rules.append(ctrl.Rule(position['N'] & velocity['Z'], force['PM']))  # Empurrar para a direita
    rules.append(ctrl.Rule(position['N'] & velocity['P'], force['Z']))   # Não empurrar

    # Linha Z
    rules.append(ctrl.Rule(position['Z'] & velocity['N'], force['PM']))  # Empurrar para a direita
    rules.append(ctrl.Rule(position['Z'] & velocity['Z'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(position['Z'] & velocity['P'], force['NM']))  # Empurrar para a esquerda

    # Linha P
    rules.append(ctrl.Rule(position['P'] & velocity['N'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(position['P'] & velocity['Z'], force['NM']))  # Empurrar para a esquerda
    rules.append(ctrl.Rule(position['P'] & velocity['P'], force['NL']))  # Empurrar forte para a esquerda

    return rules

def update_system(state, force, dt, params):
    """
    Atualiza o estado do sistema com base na força aplicada.
    """
    theta, theta_dot, x, x_dot = state
    g = params['g']
    l = params['l']
    m_p = params['m_p']
    m_c = params['m_c']
    total_mass = params['total_mass']

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_dot_sq = theta_dot ** 2

    # Equações de movimento
    temp = (force + m_p * l * theta_dot_sq * sin_theta) / total_mass
    theta_ddot = (g * sin_theta - cos_theta * temp) / (l * (4/3 - m_p * cos_theta**2 / total_mass))
    x_ddot = temp - m_p * l * theta_ddot * cos_theta / total_mass

    # Atualiza estado usando o método de Euler
    theta += theta_dot * dt
    theta_dot += theta_ddot * dt
    x += x_dot * dt
    x_dot += x_ddot * dt

    return np.array([theta, theta_dot, x, x_dot])

def plot_simulation_results(states, forces, time):
    """
    Plota os resultados da simulação.
    """
    theta = states[:, 0]
    theta_dot = states[:, 1]
    x = states[:, 2]
    x_dot = states[:, 3]

    # Cria uma figura com múltiplos subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(time, np.rad2deg(theta), label='Ângulo do Pêndulo (graus)')
    axs[0].set_ylabel('Ângulo (graus)')
    axs[0].set_title('Simulação de Controle Fuzzy do Pêndulo Invertido (GA Otimizado)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time, np.rad2deg(theta_dot), label='Velocidade Angular do Pêndulo (graus/s)', color='orange')
    axs[1].set_ylabel('Velocidade Angular (graus/s)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time, x, label='Posição do Carrinho (m)', color='green')
    axs[2].set_ylabel('Posição (m)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(time, x_dot, label='Velocidade do Carrinho (m/s)', color='red')
    axs[3].set_ylabel('Velocidade (m/s)')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(time, forces, label='Força de Controle (N)', color='purple')
    axs[4].set_xlabel('Tempo (s)')
    axs[4].set_ylabel('Força (N)')
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    genetic_algorithm()
