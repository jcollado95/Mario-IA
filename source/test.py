import retro

def main():
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()

    #buttons: [B, None, select, start, up, down, left, right, A]

    while True:
        action = env.action_space.sample()
        action = action * [0, 0, 0, 0, 0, 0, 1, 1, 1]
        obs, rew, done, info = env.step(action)
        print(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()

