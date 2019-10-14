import os
import pickle
import cv2
import numpy as np
import retro
import neat

with open('winner/winner-parallel.pkl', 'rb') as f:
    c = pickle.load(f)

#print('Loaded genome: ')
#print(c)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-parallel')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


env = retro.make(game = 'SuperMarioBros-Nes', state = 'Level1-1')

# Reset the game
ob = env.reset()
done = False

# Gets the screen size and lower its resolution
inx, iny, _ = env.observation_space.shape
inx = int(inx/8)
iny = int(iny/8)

# Create the recurrent network 
net = neat.nn.RecurrentNetwork.create(c, config)
current_max_fitness = 0
fitness_current = 0
counter = 0

while not done:
    # Renders the games screen
    env.render()

    # Apply the new resolution and put it in gray color
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))

    # Make the 2D screen a 1D array
    nn_input = np.ndarray.flatten(ob)

    # Activate the ANN
    nn_output = net.activate(nn_input)
    nn_output = [0, 0, 0, 0, 0, 0] + nn_output

    # Mario step
    ob, rew, done, info = env.step(nn_output)

    # Calculate the fitness
    fitness_current += rew
    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1
    
    # End if no improvement in 250 frames
    if counter == 250:
        done = True


print("Fitness: ", fitness_current)
