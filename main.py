import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import cv2

# Importamos a nossa central de comando!
import config

# Isso registra manualmente os jogos do Atari no Gymnasium
gym.register_envs(ale_py)

# --- 1. CRIANDO O CALLBACK (Mantido igual) ---
class CallbackDeTela(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tentativa = 1

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        if self.locals["dones"][0]:
            self.tentativa += 1
            
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        texto = f"Treinando IA - Tentativa: {self.tentativa}"
        cv2.putText(frame_bgr, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Jogo da IA", frame_bgr)
        cv2.waitKey(1)
        return True

# --- 2. O AMBIENTE ---
env = gym.make(config.NOME_DO_JOGO, render_mode="rgb_array")

# --- 3. O CÉREBRO ---
# Tentamos carregar o arquivo definido no config. Se não existir, ele vai dar erro, 
# pois no modo visual assumimos que você já treinou ou quer continuar treinando um existente.
try:
    modelo = PPO.load(config.NOME_DO_ARQUIVO_MODELO, env=env, verbose=0)
    print("Modelo carregado com sucesso!")
except:
    print("Modelo não encontrado. Criando um novo do zero...")
    modelo = PPO("CnnPolicy", env, verbose=0, ent_coef=config.ENTROPY_COEFFICIENT)


print("Iniciando o treinamento (assista a IA errando bastante)...")
callback_tela = CallbackDeTela()

# Usamos os passos do config
modelo.learn(total_timesteps=config.PASSOS_TREINO_VISUAL, callback=callback_tela)

# Salvamos usando o nome do config
modelo.save(config.NOME_DO_ARQUIVO_MODELO)

print("Treinamento concluído! Agora assista a IA jogando a sério.")

# --- 4. O TESTE FINAL ---
estado_atual, info = env.reset()
tentativa_teste = 1

for i in range(1000): 
    acao, _ = modelo.predict(estado_atual) 
    estado_atual, recompensa, bateu, truncou, info = env.step(acao)

    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    texto = f"TESTE FINAL - Tentativa: {tentativa_teste}"
    cv2.putText(frame_bgr, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    cv2.imshow("Jogo da IA", frame_bgr)
    cv2.waitKey(20) 

    if bateu or truncou:
        estado_atual, info = env.reset()
        tentativa_teste += 1

env.close()
cv2.destroyAllWindows()