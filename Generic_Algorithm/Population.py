# Genetic Algorithm, Evolving Shakespeare
import numpy as np
from DNA import dna

# A class to describe a population of virtual organisms
# In this case, each organism is just an instance of a DNA object
class population:
    
    def __init__(self, target, mutationRate, popmax):
        
        self.target = target
        self.mutationRate = mutationRate
        self.popmax = popmax

        self.dna = dna(len(target), popmax)

    def calculate_fitness(self):
        
        self.fitness = np.empty((self.popmax), dtype=bool)
        self.dna.calculate_dna_fitness(self.target)

    def natural_selection(self):

        self.dna.generate_and_mate_dna(3)

    def mutation(self):
        self.dna.mutate(self.mutationRate)

    def check(self):
        return (self.dna.total_score == len(self.target)).any()

    # Compute the current "most fit" member of the population
    def get_best(self):
        loc = np.where(self.dna.total_score == np.amax(self.dna.total_score))
        return self.dna.population_string[loc[0][0]]

    def get_average_fitness():
        pass