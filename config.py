
import ale_py
import gymnasium as gym
import sys
import signal
from stable_baselines3 import PPO
import config

# ==========================================
# CONFIGURAÇÕES GERAIS DA IA
# Altere os valores aqui e eles serão aplicados em todos os scripts
# ==========================================

# Nome do ambiente (jogo) no Gymnasium
NOME_DO_JOGO = "ALE/MsPacman-v5"

# Nome do arquivo onde o cérebro (modelo) será salvo/carregado
NOME_DO_ARQUIVO_MODELO = NOME_DO_JOGO
NOME_MODELO = "CnnPolicy"

# --- Configurações do Treino Rápido (fast.py) ---
NUMERO_DE_NUCLEOS = 4        # Quantos mundos rodar em paralelo
PASSOS_TREINO_RAPIDO = 100_000 

# --- Configurações do Treino Visual (main.py) ---
PASSOS_TREINO_VISUAL = 1_000

ENTROPY_COEFFICIENT = 0.001

# --- 3. MECANISMO DE SALVAMENTO DE EMERGÊNCIA (Ctrl + C) ---
def salvar_e_sair(modelo, env):
    print("\n\n[INTERRUPÇÃO DETECTADA]")
    print(f"Salvando o progresso atual em '{config.NOME_DO_ARQUIVO_MODELO}.zip'...")
    modelo.save(config.NOME_DO_ARQUIVO_MODELO)
    print("Modelo salvo com sucesso! Saindo agora...")
    env.close()
    sys.exit(0)

def setup(env, verbose=0):
    # Isso registra manualmente os jogos do Atari no Gymnasium
    gym.register_envs(ale_py)

    # Dizemos ao Python para chamar a nossa função quando o Ctrl+C for pressionado
    signal.signal(signal.SIGINT, salvar_e_sair)

    # --- 3. O CÉREBRO ---
    # Tentamos carregar o arquivo definido no config. Se não existir, ele vai dar erro, 
    # pois no modo visual assumimos que você já treinou ou quer continuar treinando um existente.
    try:
        modelo = PPO.load(config.NOME_DO_ARQUIVO_MODELO, env=env, verbose=verbose)
        print("Modelo carregado com sucesso!")
    except:
        print("Modelo não encontrado. Criando um novo do zero...")
        modelo = PPO(config.NOME_MODELO, env, verbose=verbose, ent_coef=config.ENTROPY_COEFFICIENT)

    return modelo