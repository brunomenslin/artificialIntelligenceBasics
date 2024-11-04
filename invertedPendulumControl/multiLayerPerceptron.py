import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from anfis import Anfis
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_training_data():
    """
    Gera dados de treinamento para o modelo ANFIS.
    Vamos simular o sistema usando o controlador fuzzy básico para coletar dados.
    """
    # Define os intervalos para as variáveis de entrada
    theta_range = np.linspace(-0.2, 0.2, 10)  # Ângulo do pêndulo
    theta_dot_range = np.linspace(-0.5, 0.5, 10)  # Velocidade angular
    x_range = np.linspace(-2, 2, 10)  # Posição do carrinho
    x_dot_range = np.linspace(-3, 3, 10)  # Velocidade do carrinho

    # Cria uma grade de todas as combinações possíveis
    theta_grid, theta_dot_grid, x_grid, x_dot_grid = np.meshgrid(
        theta_range, theta_dot_range, x_range, x_dot_range, indexing='ij'
    )

    # Achata as grades para criar vetores de entrada
    inputs = np.vstack([
        theta_grid.ravel(),
        theta_dot_grid.ravel(),
        x_grid.ravel(),
        x_dot_grid.ravel()
    ]).T

    # Calcula a força de saída usando o controlador fuzzy básico
    outputs = []
    for input_vector in inputs:
        force = basic_fuzzy_controller(input_vector)
        outputs.append(force)

    outputs = np.array(outputs)

    # Retorna entradas e saídas como dados de treinamento
    return inputs, outputs

def basic_fuzzy_controller(state):
    """
    O controlador fuzzy básico que calcula a força com base no estado.
    """
    theta, theta_dot, x, x_dot = state

    # Define variáveis fuzzy
    # Ângulo do pêndulo
    angle = ctrl.Antecedent(np.linspace(-0.2, 0.2, 401), 'angle')
    angle['N'] = fuzz.trapmf(angle.universe, [-0.2, -0.2, -0.1, 0])
    angle['Z'] = fuzz.trimf(angle.universe, [-0.1, 0, 0.1])
    angle['P'] = fuzz.trapmf(angle.universe, [0, 0.1, 0.2, 0.2])

    # Velocidade angular
    angular_velocity = ctrl.Antecedent(np.linspace(-0.5, 0.5, 101), 'angular_velocity')
    angular_velocity['N'] = fuzz.trapmf(angular_velocity.universe, [-0.5, -0.5, -0.25, 0])
    angular_velocity['Z'] = fuzz.trimf(angular_velocity.universe, [-0.05, 0, 0.05])
    angular_velocity['P'] = fuzz.trapmf(angular_velocity.universe, [0, 0.25, 0.5, 0.5])

    # Posição do carrinho
    position = ctrl.Antecedent(np.linspace(-2, 2, 401), 'position')
    position['N'] = fuzz.trapmf(position.universe, [-2, -2, -1, 0])
    position['Z'] = fuzz.trimf(position.universe, [-1, 0, 1])
    position['P'] = fuzz.trapmf(position.universe, [0, 1, 2, 2])

    # Velocidade do carrinho
    velocity = ctrl.Antecedent(np.linspace(-3, 3, 601), 'velocity')
    velocity['N'] = fuzz.trapmf(velocity.universe, [-3, -3, -1.5, 0])
    velocity['Z'] = fuzz.trimf(velocity.universe, [-0.1, 0, 0.1])
    velocity['P'] = fuzz.trapmf(velocity.universe, [0, 1.5, 3, 3])

    # Força
    force = ctrl.Consequent(np.linspace(-300, 300, 601), 'force')
    force['NL'] = fuzz.trimf(force.universe, [-300, -300, -150])
    force['NM'] = fuzz.trimf(force.universe, [-150, -75, 0])
    force['Z'] = fuzz.trimf(force.universe, [-1, 0, 1])
    force['PM'] = fuzz.trimf(force.universe, [0, 75, 150])
    force['PL'] = fuzz.trimf(force.universe, [150, 300, 300])

    # Define regras
    rules = []

    # Regras do pêndulo
    rules.append(ctrl.Rule(angle['N'] & angular_velocity['N'], force['PL']))
    rules.append(ctrl.Rule(angle['N'] & angular_velocity['Z'], force['PM']))
    rules.append(ctrl.Rule(angle['N'] & angular_velocity['P'], force['Z']))
    rules.append(ctrl.Rule(angle['Z'] & angular_velocity['N'], force['PM']))
    rules.append(ctrl.Rule(angle['Z'] & angular_velocity['Z'], force['Z']))
    rules.append(ctrl.Rule(angle['Z'] & angular_velocity['P'], force['NM']))
    rules.append(ctrl.Rule(angle['P'] & angular_velocity['N'], force['Z']))
    rules.append(ctrl.Rule(angle['P'] & angular_velocity['Z'], force['NM']))
    rules.append(ctrl.Rule(angle['P'] & angular_velocity['P'], force['NL']))

    # Regras do carrinho
    rules.append(ctrl.Rule(position['N'] & velocity['N'], force['PL']))
    rules.append(ctrl.Rule(position['N'] & velocity['Z'], force['PM']))
    rules.append(ctrl.Rule(position['N'] & velocity['P'], force['Z']))
    rules.append(ctrl.Rule(position['Z'] & velocity['N'], force['PM']))
    rules.append(ctrl.Rule(position['Z'] & velocity['Z'], force['Z']))
    rules.append(ctrl.Rule(position['Z'] & velocity['P'], force['NM']))
    rules.append(ctrl.Rule(position['P'] & velocity['N'], force['Z']))
    rules.append(ctrl.Rule(position['P'] & velocity['Z'], force['NM']))
    rules.append(ctrl.Rule(position['P'] & velocity['P'], force['NL']))

    # Cria sistema de controle e simulação
    control_system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(control_system)

    # Define entradas
    simulation.input['angle'] = np.clip(theta, -0.2, 0.2)
    simulation.input['angular_velocity'] = np.clip(theta_dot, -0.5, 0.5)
    simulation.input['position'] = np.clip(x, -2, 2)
    simulation.input['velocity'] = np.clip(x_dot, -3, 3)

    # Calcula a saída
    simulation.compute()
    force_output = simulation.output['force']

    return force_output

def train_anfis_model(inputs, outputs):
    """
    Treina um modelo ANFIS usando os dados de treinamento gerados.
    """
    # Converte dados para tensores PyTorch
    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(outputs, dtype=torch.float32).view(-1, 1)

    # Cria dataset e dataloader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define o modelo ANFIS
    model = Anfis(n_inputs=4, n_rules=81)  # 4 entradas, 3 MFs por entrada => 3^4 = 81 regras

    # Define otimizador e função de perda
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Loop de treinamento
    n_epochs = 20
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model

def simulate_pendulum_with_anfis(model):
    """
    Simula o pêndulo invertido usando o modelo ANFIS treinado.
    """
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

        # Prepara entrada para o modelo ANFIS
        input_tensor = torch.tensor([[theta, theta_dot, x, x_dot]], dtype=torch.float32)

        # Calcula a força de controle usando o modelo ANFIS
        with torch.no_grad():
            force_output = model(input_tensor).item()

        # Limita a força de saída
        force_output = np.clip(force_output, -300, 300)
        forces[i] = force_output

        # Atualiza o sistema
        state = update_system(state, force_output, dt, params)

        # Verifica se o pêndulo caiu
        if abs(theta) > np.pi / 2:
            print(f"Pêndulo caiu no tempo {i * dt:.2f}s")
            break

    # Plota resultados
    plot_simulation_results(states[:i+1], forces[:i+1], time[:i+1])

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
    axs[0].set_title('Simulação de Controle do Pêndulo Invertido com ANFIS')
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
    # Gera dados de treinamento
    inputs, outputs = generate_training_data()
    print("Dados de treinamento gerados.")

    # Treina modelo ANFIS
    anfis_model = train_anfis_model(inputs, outputs)
    print("Modelo ANFIS treinado.")

    # Simula o pêndulo invertido usando o modelo ANFIS treinado
    simulate_pendulum_with_anfis(anfis_model)
