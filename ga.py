#KARIOTIS NIKOLAOS 3243
#ORFANOUDIS KOSTAS 3303

import random
import numpy as np
from numpy.random import rand, randint

bounds = [[-1024,1023],[-1024,1023]] #define range for input
pop_size = 20

def objective_func(pair): #objective function(fitness function)
    x = pair[0]           #refers to x1
    y = pair[1]           #refers to x2
    tmp = x**2 + y        #for each solution we apply the objective function to find its function value
    return tmp

def calculate_bits_per_variable(begin,end,separation_step): #calculate number of bits needed for representation of x1(same work for x2) 
    m = end - begin                                         
    k = m * (10 ** separation_step) + 1
    n = 0
    while(2**n < k):
        n = n + 1
    return n

def decode(bounds,bits,individual):
    decoded_chromosome = list()
    largest = 2**bits
    for i in range(2):              #it decodes a chromosome(from genotype to phenotype
        start = i * bits            #2 iterations , for i=0 it decodes the first 21 bits which refer to x1 variable
        end = (i * bits) + bits     #second iteration decodes the last 21 bits which refer to x2 variable
        gene = individual[start:end] 
        chars = ''.join([str(i) for i in gene]) #convert each group of genes(x1 is one group) to a string of chars
        integer = int(chars, 2)                 #convert each group of genes to integer
        value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
        decoded_chromosome.append(value)
    return decoded_chromosome


def selection(population,fitness):              #roulette wheel selection(iss used for selecting all the individuals for the next generation)
    total_fitness = float(sum(fitness))         #sum of fuction values of all the individuals
    rel_fitness = [f/total_fitness for f in fitness] #compute ratio of individual fitness and total fitness
    probs = [sum(rel_fitness[:i+1]) for i in range(len(rel_fitness))] #probabilities of selection for each individual
    new_population = []                         
    for n in range(len(population)):
        r = rand()
        for (i,individual) in enumerate(population): #When the sum is greater than the random number generated, the corresponding individual is selected
            if(r <= probs[i]):
                new_population.append(individual)
                break
    return new_population


def crossover(parent1,parent2,crossover_rate):
    offspring = []
    if(rand() < crossover_rate):  #When the random number generated is smaller than the crossover rate then we procced to recombination 
        crossover_point = randint(1, len(parent1)-1) #A point on both parents' chromosomes is picked randomly, and designated a 'crossover point'.
        child1 = parent1[:crossover_point] + parent2[crossover_point:] #Bits to the right of that point are swapped between the two parent chromosomes. 
        child2 = parent2[:crossover_point] + parent1[crossover_point:] #This results in two offspring, each carrying some genetic information from both parents
        offspring.append(child1)
        offspring.append(child2)
    else:
        offspring.append(parent1) #else cloning the existing parents
        offspring.append(parent2)
     
    return offspring

def mutation(parent1,mutation_rate): 
    offspring = [] 
    if(rand() < mutation_rate): # When the random number generated is smaller than the mutation rate then we procced to recombination
        cut_point = randint(0, len(parent1)) # random gene
        child1 = parent1
        if(child1[cut_point] == 1):
            child1[cut_point] = 0 #flip
        else:
            child1[cut_point] = 1 #flip
 
        offspring.append(child1)
    else:
        offspring.append(parent1)


def genetic_algorithm():
    bits = calculate_bits_per_variable(-1024,1023,3) 
    population = [randint(0,2,bits*len(bounds)).tolist() for i in range(pop_size)] #initial population
    mutation_rate = 0.1
    crossover_rate = 0.6
    iterations = 10000 #number of generations to produce
    best = 0  # to keep track of best solutions(x1,x2)
    best_individual = objective_func(decode(bounds,bits,population[0])) #to keep track of best individual(one with the best y value) in each generation
    
    for gen in range(iterations): #constructs a generation
        chromosomes = []
        fitness = []
        for individual in population:
            tmp = decode(bounds,bits,individual) #genotype to phenotype
            chromosomes.append(tmp)
        for pair in chromosomes: #evaluation
            tmp = objective_func(pair)
            fitness.append(tmp)
        for i in range(len(population)):
            if(fitness[i] < best_individual): #find the minimalizers 
                best = population[i]
                best_individual = fitness[i]
                print("Generation %d, new best solution f(%s) = %f" % (gen,  chromosomes[i], fitness[i]))
        selected = selection(population,fitness) #apply roulette wheel selection
        children = list()
        if(len(selected) % 2 == 0): #if population size is even 
            for i in range(0,len(selected),2): #take pairs of individuals to apply recombination 
                p1 = selected[i]
                p2 = selected[i+1]
                crossed = crossover(p1,p2,crossover_rate) 
                for c in crossed:
                    mutation(c,mutation_rate) #apply mutation for every individual
                    children.append(c)
        else:                      #if population size is odd
            for i in range(0,len(selected),2): # get selected parents in pairs
                if(i == len(selected) - 1): #we select a random individual to produce the last pair
                    r = random.randint(0,len(selected)-1)
                    p1 = selected[i]
                    p2 = selected[r]
                else:
                     p1 = selected[i]
                     p2 = selected[i+1] 
                crossed = crossover(p1,p2,crossover_rate)
                for c in crossed:
                    mutation(c,mutation_rate)
                    children.append(c)

        population = children #current generation is updated
    return [best,best_individual]


res = genetic_algorithm()
bits = calculate_bits_per_variable(-1024,1023,3)
decoded = decode(bounds,bits,res[0])
print('FINAL RESULT : f(%s) = %f' % (decoded, res[1]))
