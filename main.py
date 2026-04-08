import gymnasium as gym
from stable_baselines3 import PPO

# 1. O Ambiente: Criamos o jogo. 
# O render_mode="human" é o que faz a janela gráfica abrir para você assistir.
env = gym.make("CartPole-v1", render_mode="human")

# 2. O Cérebro: Criamos a IA usando o algoritmo PPO.
# MlpPolicy significa que ela usará uma Rede Neural padrão.
modelo = PPO("MlpPolicy", env, verbose=1)

print("Iniciando o treinamento (assista a IA errando bastante)...")
# 3. O Treino: Deixamos ela tentar 10.000 vezes. 
modelo.learn(total_timesteps=10000)

print("Treinamento concluído! Agora assista a IA jogando a sério.")
# 4. O Teste Final: Resetamos o jogo para ver o que ela aprendeu
estado_atual, info = env.reset()

for i in range(1000): # Roda 1000 quadros de animação
    # A IA analisa o estado atual e decide a melhor ação (ir para direita ou esquerda)
    acao, _ = modelo.predict(estado_atual) 
    
    # Executamos a ação no jogo e recebemos o resultado
    estado_atual, recompensa, bateu, truncou, info = env.step(acao)

    # Se ela deixou o bastão cair, reseta para tentar de novo
    if bateu or truncou:
        estado_atual, info = env.reset()

env.close()