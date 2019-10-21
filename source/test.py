import retro
import numpy as np

def main():
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()

    #buttons: [B, None, select, start, up, down, left, right, A]

    while True:
        input = obs.flatten()

        output = env.action_space.sample()
        output = np.concatenate([np.zeros(6), output[-3:]])
        
        obs, rew, done, info = env.step(output)

        print(output)
        
        env.render()
        
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()

