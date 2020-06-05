# Genetic Algorithm, Evolving Shakespeare

# Demonstration of using a genetic algorithm to perform a search
# setup()
  # Step 1: The populationation 
    # Create an empty populationation (an array or ArrayList)
    # Fill it with DNA encoded objects (pick random values to start)
# draw()
  # Step 1: Selection 
    # Create an empty mating pool (an empty ArrayList)
    # For every member of the populationation, evaluate its fitness based on some criteria / function, 
    #  and add it to the mating pool in a manner consistant with its fitness, i.e. the more fit it 
    #  is the more times it appears in the mating pool, in order to be more likely picked for reproduction.
  # Step 2: Reproduction Create a new empty populationation
    # Fill the new populationation by executing the following steps:
    #   1. Pick two "parent" objects from the mating pool.
    #   2. Crossover -- create a "child" object by mating these two parents.
    #   3. Mutation -- mutate the child's DNA based on a given probability.
    #   4. Add the child object to the new populationation.
    # Replace the old populationation with the new populationation
  
   # Rinse and repeat

from Population import population

string = "my name is tunc"
popmax = 200
mutationRate = 0.05

pop = population(string, mutationRate, popmax)

goal = False
count = 0
best_guess_list = []

while True:
  count += 1

  pop.calculate_fitness()
  pop.natural_selection()
  best_guess_list.append(pop.get_best())
  if pop.check():
    break
  pop.mutation()

print("Number of iterations required to find the string is: {}".format(count))
print(best_guess_list)