import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle
import os           # directory and file paths
import argparse	    # Input arguments

env = retro.make(game = 'SuperMarioBros-Nes', state = 'Level1-1')

def eval_genome(genome, config):
    # Reset the game
    ob = env.reset()
    done = False

    # Gets the screen size and lower its resolution
    inx, iny, _ = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)

    # Create the recurrent network 
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    max_fitness = 0
    current_fitness = 0
    counter = 0

    while not done:
        # Renders the games screen
        #env.render()

        # Apply the new resolution and put it in gray color
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))

        # Make the 2D screen a 1D array
        nn_input = ob.flatten()

        # Activate the ANN
        nn_output = net.activate(nn_input)
        nn_output = np.concatenate([np.zeros(6), np.asarray(nn_output)])

        # Mario step
        obs, rew, done, info = env.step(nn_output)
        ob = obs

        # Calculate the fitness
        current_fitness += rew
        if current_fitness > max_fitness:
            max_fitness = current_fitness
            counter = 0
        else:
            counter += 1
        
        # End if no improvement in 250 frames
        if counter == 250:
            done = True

    genome.fitness = max_fitness
    print("Fitness: ", genome.fitness)
    return genome.fitness


def run(config_file, it, cores):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)
    
    # Create the population
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for x generations
    pe = neat.ParallelEvaluator(cores, eval_genome)
    winner = p.run(pe.evaluate, it)

    # Display and save the winning genome
    print('\nBest genome:\n{!s}'.format(winner))
    with open('winner/winner-parallel.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations", default=100, help="Número de iteraciones del algoritmo", type=int)
    parser.add_argument("-c", "--cores", default=4, help="Número de procesadores paralelos", type=int)

    args = parser.parse_args()
    if args.iterations:
        it = args.iterations
        print(it)
    if args.cores:
        cores = args.cores
        print(cores)

    # Determine path to config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-parallel')
    run(config_path, it, cores)
