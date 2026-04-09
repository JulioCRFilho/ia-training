import os
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import config

# --- 1. CONFIGURAÇÃO DO AMBIENTE ---
env = make_vec_env(config.NOME_DO_JOGO, n_envs=config.NUMERO_DE_NUCLEOS)

# --- 2. O CÉREBRO ---
if os.path.exists(f"{config.NOME_DO_ARQUIVO_MODELO}.zip"):
    print(f"Modelo {config.NOME_DO_ARQUIVO_MODELO} encontrado! Carregando...")
    modelo = PPO.load(config.NOME_DO_ARQUIVO_MODELO, env=env, verbose=1) 
else:
    print(f"Criando IA do zero para {config.NOME_DO_JOGO}...")
    modelo = PPO(config.NOME_MODELO, env, verbose=1) # CarRacing usa CnnPolicy por causa das imagens

# --- 3. MECANISMO DE SALVAMENTO DE EMERGÊNCIA (Ctrl + C) ---
def salvar_e_sair(sig, frame):
    print("\n\n[INTERRUPÇÃO DETECTADA]")
    print(f"Salvando o progresso atual em '{config.NOME_DO_ARQUIVO_MODELO}.zip'...")
    modelo.save(config.NOME_DO_ARQUIVO_MODELO)
    print("Modelo salvo com sucesso! Saindo agora...")
    env.close()
    sys.exit(0)

# Dizemos ao Python para chamar a nossa função quando o Ctrl+C for pressionado
signal.signal(signal.SIGINT, salvar_e_sair)

# --- 4. O TREINAMENTO ---
print(f"Iniciando treinamento paralelo. Pressione Ctrl+C a qualquer momento para salvar e sair.")
try:
    modelo.learn(total_timesteps=config.PASSOS_TREINO_RAPIDO)
    # Se terminar naturalmente, salva também
    modelo.save(config.NOME_DO_ARQUIVO_MODELO)
    print("Treinamento finalizado e salvo!")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
finally:
    env.close()