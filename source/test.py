import retro

def main():
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()

    #buttons: [B, None, select, start, up, down, left, right, A]
    action1 = [0,0,0,0,0,0,0,1,1]
    action2 = [0,1,0,0,0,0,0,0,0]
    #action = env.action_space.sample()

    it = False
    while True:
        if it:
            obs, rew, done, info = env.step(action1)
        else:
            obs, rew, done, info = env.step(action2)
        it = not it

        print(action1)
        print(action2)
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()

