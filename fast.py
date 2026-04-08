import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Importamos a nossa central de comando!
import config

# --- 1. MÚLTIPLOS AMBIENTES (O Multiverso) ---
env = make_vec_env(config.NOME_DO_JOGO, n_envs=config.NUMERO_DE_NUCLEOS)

# --- 2. O CÉREBRO ---
if os.path.exists(f"{config.NOME_DO_ARQUIVO_MODELO}.zip"):
    print(f"Modelo {config.NOME_DO_ARQUIVO_MODELO} encontrado! Carregando...")
    modelo = PPO.load(config.NOME_DO_ARQUIVO_MODELO, env=env, verbose=1) 
else:
    print(f"Criando IA para treinar em {config.NUMERO_DE_NUCLEOS} mundos paralelos...")
    modelo = PPO("MlpPolicy", env, verbose=1)

# --- 3. O TREINAMENTO PARALELO ---
print(f"Iniciando treinamento paralelo de {config.PASSOS_TREINO_RAPIDO} passos...")
modelo.learn(total_timesteps=config.PASSOS_TREINO_RAPIDO)

# --- 4. SALVANDO O RESULTADO ---
modelo.save(config.NOME_DO_ARQUIVO_MODELO)
print("Treinamento finalizado e modelo salvo com sucesso!")

env.close()