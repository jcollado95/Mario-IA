import retro
import numpy as np
import cv2 # For image reduction
import neat
import pickle

env = retro.make(game='SuperMarioBros-Nes', state='Level1-1')
oned_image = []

def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        
        # Reset the game
        ob = env.reset()
        done = False

        # Gets the screen size
        inx,iny,_ = env.observation_space.shape

        # Lower screen resolution
        inx = int(inx/8)
        iny = int(iny/8)

        # Create the recurrent network
        net = neat.nn.RecurrentNetwork.create(genome,config)
        current_max_fitness = 0
        fitness_current = 0
        counter = 0

        while not done:
            # Renders the game's screen
            #env.render()
            
            # Lower the resolution and get the gray color
            ob = cv2.resize(ob,(inx,iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob,(inx,iny))
            
            # Make the 2D screen a 1D array
            oned_image = np.ndarray.flatten(ob)

            # Activate the NN and use the output as the Mario action
            nn_output = net.activate(oned_image)
            ob, rew, done, info = env.step(nn_output)
            
            # Calculate the fitness
            fitness_current += rew
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter+=1

            # Train until mario doesnt get better for 250 consecutive frames
            if counter == 250:
                done = True
            
            genome.fitness = fitness_current

        print("ID: ", genome_id, "Fitness: ", fitness_current)


# Set up the NN configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
# Create the population
p = neat.Population(config)

# Statistics
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Save the process after each 10 frames
p.add_reporter(neat.Checkpointer(10))

# Save the winner in a file
winner = p.run(eval_genomes)
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)