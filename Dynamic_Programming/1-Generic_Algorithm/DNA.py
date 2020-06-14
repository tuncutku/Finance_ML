import random
import string
import numpy as np
from math import ceil

class dna():
    def __init__(self, n_characters, n_population):

        self.dim = (n_population, n_characters)
        self.population_array = np.empty(self.dim, dtype=str)
        self.population_string = []

        for idx in range(n_population):

            self.population_array[idx] = random.choices(population=string.printable, k=n_characters)
            self.population_string.append("".join(map(str, self.population_array[idx])))

    def calculate_dna_fitness(self, string):

        self.fitness_mat = np.empty(self.dim, dtype=bool)

        string_list = np.tile(np.array(list(string)), (self.dim[0], 1))
        fitness_list = (self.population_array == string_list) * 1
        self.total_score = np.sum(fitness_list, axis=1)
        exp_score = map(lambda score: np.exp(score), self.total_score)
        self.points = list(exp_score)

    @staticmethod
    def string_to_array(string):

        array = []
        for idx in range(len(string)):
            array.append(string[idx])
        return array
    
    @staticmethod
    def array_to_string(array):

        string = []
        string.append("".join(map(str, array)))
        return string

    def generate_and_mate_dna(self, n_combination):
        
        normalized_p = self.points / np.sum(self.points)
        
        division = self.dim[1]/n_combination
        loop_order = list(range(n_combination))*ceil(division)
        
        for idx in range(self.dim[0]):
            mate_array = np.random.choice(self.population_string, n_combination, True, normalized_p)
            jdx = 0
            new_string = ""
            while jdx < self.dim[1]: 
                new_string += mate_array[loop_order[jdx]][jdx]
                jdx += 1
            self.population_string[idx] = new_string
            self.population_array[idx] = self.string_to_array(self.population_string[idx])

    def mutate(self, mutationRate):
        
        new_population_string = []
        np.random.seed(42)
        for idx in range(self.dim[0]):
            for jdx in range(self.dim[1]):
                if np.random.rand() <= mutationRate:
                    self.population_array[idx, jdx] = random.choices(population=string.printable, k=1)[0]
            new_population_string.append("".join(map(str, self.population_array[idx])))
        self.population_string = new_population_string

    

    