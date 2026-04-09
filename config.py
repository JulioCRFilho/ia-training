# ==========================================
# CONFIGURAÇÕES GERAIS DA IA
# Altere os valores aqui e eles serão aplicados em todos os scripts
# ==========================================

# Nome do ambiente (jogo) no Gymnasium
NOME_DO_JOGO = "ALE/MsPacman-v5"

# Nome do arquivo onde o cérebro (modelo) será salvo/carregado
NOME_DO_ARQUIVO_MODELO = NOME_DO_JOGO

# --- Configurações do Treino Rápido (fast.py) ---
NUMERO_DE_NUCLEOS = 4        # Quantos mundos rodar em paralelo
PASSOS_TREINO_RAPIDO = 1_000_000 

# --- Configurações do Treino Visual (main.py) ---
PASSOS_TREINO_VISUAL = 1_000

ENTROPY_COEFFICIENT = 0.001