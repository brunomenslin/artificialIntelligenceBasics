import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def define_pendulum_membership_functions():
    """
    Define as funções de pertinência para as variáveis do pêndulo.
    """
    # Ângulo do pêndulo em radianos (-0.2 a 0.2)
    angle = ctrl.Antecedent(np.linspace(-0.2, 0.2, 401), 'angle')
    # Velocidade angular do pêndulo em radianos por segundo (-0.2 a 0.2)
    angular_velocity = ctrl.Antecedent(np.linspace(-0.2, 0.2, 401), 'angular_velocity')
    # Força de saída do controlador do pêndulo (-200 a 200 Newtons)
    force = ctrl.Consequent(np.linspace(-200, 200, 401), 'force', defuzzify_method='centroid')

    # Funções de pertinência do ângulo do pêndulo
    angle['N'] = fuzz.trapmf(angle.universe, [-0.2, -0.2, -0.1, 0])
    angle['Z'] = fuzz.trapmf(angle.universe, [-0.1, -0.03, 0.03, 0.1])
    angle['P'] = fuzz.trapmf(angle.universe, [0, 0.1, 0.2, 0.2])

    # Funções de pertinência da velocidade angular do pêndulo
    angular_velocity['N'] = fuzz.trapmf(angular_velocity.universe, [-0.2, -0.2, -0.1, 0])
    angular_velocity['Z'] = fuzz.trapmf(angular_velocity.universe, [-0.15, -0.03, 0.03, 0.15])
    angular_velocity['P'] = fuzz.trapmf(angular_velocity.universe, [0, 0.1, 0.2, 0.2])

    # Funções de pertinência da força
    force['NL'] = fuzz.trimf(force.universe, [-200, -200, -100])
    force['NM'] = fuzz.trimf(force.universe, [-100, -80, -40])
    force['NS'] = fuzz.trimf(force.universe, [-10, -5, 0])
    force['Z'] = fuzz.trimf(force.universe, [-1, 0, 1])
    force['PS'] = fuzz.trimf(force.universe, [0, 5, 10])
    force['PM'] = fuzz.trimf(force.universe, [40, 80, 100])
    force['PL'] = fuzz.trimf(force.universe, [100, 200, 200])

    return angle, angular_velocity, force

def define_pendulum_rules(angle, angular_velocity, force):
    """
    Define regras fuzzy para o controlador do pêndulo.
    """
    rules = []

    # Matriz de regras para ângulo e velocidade angular do pêndulo
    # Linha N
    rules.append(ctrl.Rule(angle['N'] & angular_velocity['N'], force['PL']))  # Empurrar forte para esquerda
    rules.append(ctrl.Rule(angle['N'] & angular_velocity['Z'], force['PM']))  # Empurrar para esquerda
    rules.append(ctrl.Rule(angle['N'] & angular_velocity['P'], force['Z']))   # Não empurrar

    # Linha Z
    rules.append(ctrl.Rule(angle['Z'] & angular_velocity['N'], force['PS']))  # Empurrar levemente para esquerda
    rules.append(ctrl.Rule(angle['Z'] & angular_velocity['Z'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(angle['Z'] & angular_velocity['P'], force['NS']))  # Empurrar levemente para direita

    # Linha P
    rules.append(ctrl.Rule(angle['P'] & angular_velocity['N'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(angle['P'] & angular_velocity['Z'], force['NM']))  # Empurrar para direita
    rules.append(ctrl.Rule(angle['P'] & angular_velocity['P'], force['NL']))  # Empurrar forte para direita

    return rules

def define_cart_membership_functions():
    """
    Define as funções de pertinência para as variáveis do carrinho.
    """
    # Posição do carrinho em metros (-2 a 2)
    position = ctrl.Antecedent(np.linspace(-2, 2, 401), 'position')
    # Velocidade linear do carrinho em m/s (-3 a 3)
    velocity = ctrl.Antecedent(np.linspace(-3, 3, 601), 'velocity')
    # Força de saída do controlador do carrinho (-100 a 100 Newtons)
    force = ctrl.Consequent(np.linspace(-100, 100, 201), 'force', defuzzify_method='centroid')

    # Funções de pertinência da posição do carrinho
    position['N'] = fuzz.trapmf(position.universe, [-2, -2, -1.5, -0.5])
    position['Z'] = fuzz.trapmf(position.universe, [-1.5, -0.5, 0.5, 1.5])
    position['P'] = fuzz.trapmf(position.universe, [0.5, 1.5, 2, 2])

    # Funções de pertinência da velocidade linear do carrinho
    velocity['N'] = fuzz.trapmf(velocity.universe, [-3, -3, -1.5, 0])
    velocity['Z'] = fuzz.trapmf(velocity.universe, [-1.5, -0.5, 0.5, 1.5])
    velocity['P'] = fuzz.trapmf(velocity.universe, [0, 1.5, 3, 3])

    # Funções de pertinência da força
    force['NL'] = fuzz.trimf(force.universe, [-100, -100, -50])
    force['NM'] = fuzz.trimf(force.universe, [-10, -5, 0])
    force['NS'] = fuzz.trimf(force.universe, [-2, -1, 0])
    force['Z'] = fuzz.trimf(force.universe, [-1, 0, 1])
    force['PS'] = fuzz.trimf(force.universe, [0, 1, 2])
    force['PM'] = fuzz.trimf(force.universe, [0, 5, 10])
    force['PL'] = fuzz.trimf(force.universe, [50, 100, 100])

    return position, velocity, force

def define_cart_rules(position, velocity, force):
    """
    Define regras fuzzy para o controlador do carrinho.
    """
    rules = []

    # Matriz de regras para posição e velocidade linear do carrinho
    # Linha N
    rules.append(ctrl.Rule(position['N'] & velocity['N'], force['PL']))  # Empurrar forte para direita
    rules.append(ctrl.Rule(position['N'] & velocity['Z'], force['PM']))  # Empurrar para direita
    rules.append(ctrl.Rule(position['N'] & velocity['P'], force['Z']))   # Não empurrar

    # Linha Z
    rules.append(ctrl.Rule(position['Z'] & velocity['N'], force['PS']))  # Empurrar levemente para direita
    rules.append(ctrl.Rule(position['Z'] & velocity['Z'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(position['Z'] & velocity['P'], force['NS']))  # Empurrar levemente para esquerda

    # Linha P
    rules.append(ctrl.Rule(position['P'] & velocity['N'], force['Z']))   # Não empurrar
    rules.append(ctrl.Rule(position['P'] & velocity['Z'], force['NM']))  # Empurrar para esquerda
    rules.append(ctrl.Rule(position['P'] & velocity['P'], force['NL']))  # Empurrar forte para esquerda

    return rules

def plot_membership_functions():
    """
    Plota as funções de pertinência para todas as variáveis na mesma figura.
    """
    # Variáveis do pêndulo
    angle, angular_velocity, pendulum_force = define_pendulum_membership_functions()
    # Variáveis do carrinho
    position, velocity, cart_force = define_cart_membership_functions()

    # Cria uma figura com múltiplos subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Plota funções de pertinência do ângulo do pêndulo
    ax = axs[0, 0]
    for term in angle.terms:
        ax.plot(angle.universe, angle[term].mf, label=term)
    ax.set_title('Ângulo do Pêndulo (rad)')
    ax.set_xlabel('Ângulo (rad)')
    ax.set_ylabel('Grau de Pertinência')
    ax.legend()
    ax.grid(True)

    # Plota funções de pertinência da velocidade angular do pêndulo
    ax = axs[0, 1]
    for term in angular_velocity.terms:
        ax.plot(angular_velocity.universe, angular_velocity[term].mf, label=term)
    ax.set_title('Velocidade Angular do Pêndulo (rad/s)')
    ax.set_xlabel('Velocidade Angular (rad/s)')
    ax.set_ylabel('Grau de Pertinência')
    ax.legend()
    ax.grid(True)

    # Plota funções de pertinência da força do pêndulo
    ax = axs[1, 0]
    for term in pendulum_force.terms:
        ax.plot(pendulum_force.universe, pendulum_force[term].mf, label=term)
    ax.set_title('Força do Controlador do Pêndulo (N)')
    ax.set_xlabel('Força (N)')
    ax.set_ylabel('Grau de Pertinência')
    ax.legend()
    ax.grid(True)

    # Plota funções de pertinência da posição do carrinho
    ax = axs[1, 1]
    for term in position.terms:
        ax.plot(position.universe, position[term].mf, label=term)
    ax.set_title('Posição do Carrinho (m)')
    ax.set_xlabel('Posição (m)')
    ax.set_ylabel('Grau de Pertinência')
    ax.legend()
    ax.grid(True)

    # Plota funções de pertinência da velocidade do carrinho
    ax = axs[2, 0]
    for term in velocity.terms:
        ax.plot(velocity.universe, velocity[term].mf, label=term)
    ax.set_title('Velocidade do Carrinho (m/s)')
    ax.set_xlabel('Velocidade (m/s)')
    ax.set_ylabel('Grau de Pertinência')
    ax.legend()
    ax.grid(True)

    # Plota funções de pertinência da força do carrinho
    ax = axs[2, 1]
    for term in cart_force.terms:
        ax.plot(cart_force.universe, cart_force[term].mf, label=term)
    ax.set_title('Força do Controlador do Carrinho (N)')
    ax.set_xlabel('Força (N)')
    ax.set_ylabel('Grau de Pertinência')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_force(state, pendulum_ctrl_sim, cart_ctrl_sim):
    """
    Calcula a força total a ser aplicada com base no estado atual.
    """
    theta, theta_dot, x, x_dot = state

    # Entradas do controlador do pêndulo
    angle_input = theta  # Já em radianos
    angular_velocity_input = theta_dot  # Radianos por segundo

    # Entradas do controlador do carrinho
    position_input = x  # Metros
    velocity_input = x_dot  # Metros por segundo

    # Controlador do pêndulo
    pendulum_ctrl_sim.input['angle'] = np.clip(angle_input, -0.2, 0.2)
    pendulum_ctrl_sim.input['angular_velocity'] = np.clip(angular_velocity_input, -0.2, 0.2)
    pendulum_ctrl_sim.compute()
    force_pendulum = pendulum_ctrl_sim.output['force']

    # Controlador do carrinho
    cart_ctrl_sim.input['position'] = np.clip(position_input, -2, 2)
    cart_ctrl_sim.input['velocity'] = np.clip(velocity_input, -3, 3)
    cart_ctrl_sim.compute()
    force_cart = cart_ctrl_sim.output['force']

    # Força total é a soma (você pode escolher ponderá-las diferentemente)
    total_force = force_pendulum + force_cart

    # Limita a força total
    total_force = np.clip(total_force, -300, 300)

    return total_force

def update_system(state, force, dt, params):
    """
    Atualiza o estado do sistema com base na força aplicada.
    """
    theta, theta_dot, x, x_dot = state
    g = params['g']
    l = params['l']
    m_p = params['m_p']
    m_c = params['m_c']
    total_mass = m_p + m_c

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    theta_dot_sq = theta_dot**2

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

def plot_results(states, forces, time):
    """
    Plota os resultados da simulação na mesma figura.
    """
    theta = states[:, 0]
    theta_dot = states[:, 1]
    x = states[:, 2]
    x_dot = states[:, 3]

    # Cria uma figura com múltiplos subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    axs[0].plot(time, np.rad2deg(theta), label='Ângulo do Pêndulo (graus)')
    axs[0].set_ylabel('Ângulo (graus)')
    axs[0].set_title('Simulação do Controle Fuzzy do Pêndulo Invertido')
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

def simulate_pendulum():
    """
    Simula o pêndulo invertido usando os controladores fuzzy.
    """
    # Define funções de pertinência e regras do pêndulo
    angle, angular_velocity, pendulum_force = define_pendulum_membership_functions()
    pendulum_rules = define_pendulum_rules(angle, angular_velocity, pendulum_force)
    pendulum_ctrl = ctrl.ControlSystem(pendulum_rules)
    pendulum_ctrl_sim = ctrl.ControlSystemSimulation(pendulum_ctrl)

    # Define funções de pertinência e regras do carrinho
    position, velocity, cart_force = define_cart_membership_functions()
    cart_rules = define_cart_rules(position, velocity, cart_force)
    cart_ctrl = ctrl.ControlSystem(cart_rules)
    cart_ctrl_sim = ctrl.ControlSystemSimulation(cart_ctrl)

    # Plota funções de pertinência
    plot_membership_functions()

    # Parâmetros da simulação
    dt = 0.01  # Passo de tempo (segundos)
    t_max = 10  # Tempo total de simulação (segundos)
    num_steps = int(t_max / dt)  # Número de passos da simulação
    time = np.linspace(0, t_max, num_steps)  # Array de tempo

    # Parâmetros físicos do sistema do pêndulo invertido
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

        # Calcula força
        force = calculate_force(state, pendulum_ctrl_sim, cart_ctrl_sim)
        forces[i] = force

        # Atualiza sistema
        state = update_system(state, force, dt, params)

    # Plota resultados
    plot_results(states, forces, time)

if __name__ == "__main__":
    simulate_pendulum()
