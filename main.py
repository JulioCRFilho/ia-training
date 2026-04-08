import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import cv2

# --- 1. CRIANDO O CALLBACK (O Interceptador para desenhar na tela) ---
class CallbackDeTela(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.tentativa = 1 # Variável de controle de tentativa

    def _on_step(self) -> bool:
        # Pega o ambiente do jogo
        env = self.training_env.envs[0]
        
        # O SB3 avisa quando o episódio termina (o bastão caiu), então somamos 1 na tentativa
        if self.locals["dones"][0]:
            self.tentativa += 1
            
        # Pega a "foto" atual do jogo
        frame = env.render()
        
        # O Gym usa cores no formato RGB, mas o OpenCV usa BGR. Precisamos converter:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Prepara o texto (Texto, posição X e Y, fonte, escala, cor BGR, espessura)
        texto = f"Treinando IA - Tentativa: {self.tentativa}"
        cv2.putText(frame_bgr, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Exibe a imagem numa janela
        cv2.imshow("Jogo da IA", frame_bgr)
        cv2.waitKey(1) # Atualiza a tela super rápido
        
        return True

# --- 2. O AMBIENTE ---
# Alteramos de "human" para "rgb_array" para podermos manipular a imagem
env = gym.make("CartPole-v1", render_mode="rgb_array")

# --- 3. O CÉREBRO ---
modelo = PPO.load("meu_cerebro_cartpole", env=env, verbose=0)

print("Iniciando o treinamento (assista a IA errando bastante)...")

# Passamos nosso Callback para o modelo usar enquanto treina
callback_tela = CallbackDeTela()
modelo.learn(total_timesteps=10000, callback=callback_tela)
modelo.save("meu_cerebro_cartpole")

print("Treinamento concluído! Agora assista a IA jogando a sério.")

# --- 4. O TESTE FINAL ---
estado_atual, info = env.reset()
tentativa_teste = 1

for i in range(1000): # Roda 1000 quadros de animação
    acao, _ = modelo.predict(estado_atual) 
    estado_atual, recompensa, bateu, truncou, info = env.step(acao)

    # Pegamos a imagem e desenhamos na tela igual fizemos no Callback
    frame = env.render()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    texto = f"TESTE FINAL - Tentativa: {tentativa_teste}"
    cv2.putText(frame_bgr, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    cv2.imshow("Jogo da IA", frame_bgr)
    cv2.waitKey(20) # 20 milissegundos para ficar numa velocidade assistível

    if bateu or truncou:
        estado_atual, info = env.reset()
        tentativa_teste += 1

# Fecha o ambiente e fecha as janelas do OpenCV
env.close()
cv2.destroyAllWindows()