import retro
import numpy as np

class Action():
        def __init__(self):
                self.actionMask = [
                        [0,1,0,0,1,1,1,1,1]                       
                ]
        
        def getRandomAction(self):
                action = np.random.choice(2,9)
                return np.ndarray.flatten(action * self.actionMask)

def main():
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()
    while True:
        action = Action().getRandomAction()
        print(action)
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
        main()


