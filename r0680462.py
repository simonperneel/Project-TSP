from numpy.core.fromnumeric import mean
import Reporter
import numpy as np
import random
import statistics
import scipy.stats as stats
import math
import time


class Individual:
    def __init__(self, tour):
        self.tour = tour
        self.alpha = 0.5  # probability of mutation
        self.prc = 0.99  # probability of recombination
        self.pcw = 0.5   # probability of crowding

class Params:

    # set start parameters
    def __init__(self, distanceMatrix):
        self.popsize = 250  # population size
        self.amountOfOffspring = 150  # amount of trials to generate a child (see prc)
        self.k = 5  # for k-tournament selection
        self.distanceMatrix = distanceMatrix  # matrix with the cost between cities
        self.pheur = 0.5  # amount of the pop that is initialized with nn heuristic

class r0680462:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def cost(self, ind, distanceMatrix):
        tour = ind.tour
        cost = distanceMatrix[tour[0]][tour[len(tour)-1]]  # cost between first and last city to make circle complete
        for i in range(len(tour) - 1):
            cost += distanceMatrix[tour[i]][tour[i+1]]
        return cost

    def distance(self, city1, city2, params):
        distance = params.distanceMatrix[city1][city2]
        return distance

    # amounts of swaps (not minimum) that tours are away from each other, used as 'distance' between individuals
    def swap_distance(self, ind1, ind2):
        swaps = 0
        tour1 = ind1.tour.tolist()
        tour2 = ind2.tour.tolist()
        for i in range(len(tour1)):
            if tour1[i] != tour2[i]:
                swaps += 1
                element = tour1[i]
                index = tour1.index(tour2[i])  # find index of correct element
                tour1[index] = element  # swap
                tour1[i] = tour2[i]
        return swaps

    def hamming_distance(self, ind1, ind2):
        distance = 0
        for i in range(len(ind1.tour)):
            if ind1.tour[i] != ind2.tour[i]:
                distance += 1
        return distance

    """ calculate kendall tau distance, a correlation measure for lists
        Value close to 1: strong agreement, close to 0: strong disagreement """
    def kendalltau_distance(self, ind1, ind2):
        distance, pvalue = stats.kendalltau(ind1.tour, ind2.tour)
        return distance

    """ returns closest individual of a sample from the population to a given individual """
    def closestIndividual(self, ind, population):
        all_distances = []
        for i in range(len(population) - 1):
            all_distances.append(self.hamming_distance(ind, population[i]))
            index = all_distances.index(min(all_distances))
        # closest to sampled individuals
        #sampled_ind = []
        #for i in range(int(len(population)-1)):  # sample individuals
        #     sampled_ind.append(random.randint(0, len(population) - 1))  # fill list with indices of the sampled inds
        #     all_distances.append(self.hamming_distance(ind, population[sampled_ind[i]]))
        #   index = all_distances.index(min(all_distances))  # index of closest individual

        return population[index], index

    def init(self, params, nlen):
        population = []
        # random initialization
        for i in range(int(params.popsize*(1-params.pheur))):
            tour = np.random.permutation(nlen)
            ind = Individual(tour)
            population.append(ind)
        # initialize % of pop with heuristic good individuals
        for i in range(int(params.popsize*params.pheur)):
            ind = self.init_nn(params, nlen)
            population.append(ind)
        return population

    """ initializes an individual with nearest neighbour"""
    def init_nn(self, params, nlen):
        start = np.random.choice(range(nlen))  # random pick first city
        cities = set(range(nlen))  # all cities
        tour = [start]
        unvisited = set(cities - {start})  # unvisited cities)
        unvisited = list(unvisited)
        while unvisited:
            C = self.nearestneighbor(tour[-1], unvisited, params)
            tour.append(C)
            unvisited.remove(C)
        tour = np.array(tour)
        ind = Individual(tour)
        return ind

    """ find the city that is nearest to city A """
    def nearestneighbor(self, A, cities, params):
        return min(cities, key=lambda c: self.distance(c,A, params))

    """ k-tournament selection """
    def selection(self, population, params):
        inds = []
        for i in range(params.k):
            r = random.randint(0, len(population) - 1)
            inds.append(r)
        all_costs = []
        for i in range(params.k):
            all_costs.append(self.cost(population[inds[i]], params.distanceMatrix))
        index = all_costs.index(min(all_costs))
        return population[inds[index]], inds[index]

    """ sample k individuals from population """
    def sample(self, population, k):
        inds = []
        for i in range(k):
            r = random.randint(0, len(population) - 1)
            inds.append(population[r])
        return


    def swap_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while (i1 == i2):
                i2 = random.randint(0, nlen - 1)

            tour = individual.tour
            tour[i1], tour[i2] = tour[i2], tour[i1]
            individual.tour = tour

    """ reverse sequence mutation reverses a random selected part of the tour"""
    def rs_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            start = random.randint(0, nlen -1)
            end = start
            while start == end:
                end = random.randint(0, nlen - 1)
            if start > end:
                start, end = end, start
            tour = individual.tour
            tour[start:end+1] = np.flip(tour[start:end+1])  # reverse part of tour


    """ randomly selects two cities and inserts one before the other """
    def ordered_mutation(self, individual, nlen):
        if random.random() < individual.alpha:  # alpha % chance of mutation
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            tour = individual.tour
            tmp = tour[i1]
            tour = np.delete(tour, i1)
            tour = np.insert(tour, i2, tmp)
            individual.tour = tour

    """"" crossover function """
    def ordered_crossover(self, p1, p2, offspring, nlen):
        if random.random() < p1.prc:  # prc % chance of recombination
            # start index for subset that will be transferred to the child
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)  # end index of subset
            if i1 > i2:
                i1, i2 = i2, i1  # start index < end index

            child_tour = np.full(nlen, -1, dtype=int)
            for i in range(i1, i2):  # copy subset of p1 to child
                child_tour[i] = p1.tour[i]

            # delete values that are already in child from p2
            p2_tour = p2.tour
            values_to_delete = p1.tour[i1:i2]  # values to delete from p2
            for i in range(len(values_to_delete)):
                # removes element values_to_delete[i] from p2
                p2_tour = p2_tour[p2_tour != values_to_delete[i]]

            # insert remaining values of p2 into child
            j = 0
            for i in range(len(child_tour)):
                if child_tour[i] == -1:  # empty spot
                    child_tour[i] = p2_tour[j]
                    j += 1

            # create child
            child = Individual(child_tour)
            # add child to offspring
            offspring.append(child)

        else:  # no recombination happened
            pass

    """ only keep the best idividuals in the population, eliminate others """
    def elimination(self, population, params):
        population = sorted(population, key=lambda x: self.cost(x, params.distanceMatrix))
        return population[:params.popsize]

    """ crowding with k-tournament promotion """
    def crowding(self, population, params):
        next_gen = []
        # promote ind to next generation by k-tournament
        for i in range(params.popsize):
            promoted_ind, index = self.selection(population, params)
            next_gen.append(promoted_ind)  # add to next generation
            population.pop(index)
            if random.random() < promoted_ind.pcw:  # probability of crowding close individual
                closestInd, index = self.closestIndividual(promoted_ind, population)  # find closest individual from promoted one
                population.pop(index)

        return next_gen

    """ crowding with elitism promotion """
    def crowding2(self, population, params):
        next_gen = []
        # promote ind to next generation by elitism
        population = sorted(population, key=lambda x: self.cost(x, params.distanceMatrix))  # rank population based on fitness
        for i in range(params.popsize):
            promoted_ind = population[i]
            next_gen.append(promoted_ind)  # copy to next generation
            population.pop(i)  # remove promoted ind from population
            if random.random() < population[i].pcw:
                closestInd, index = self.closestIndividual(promoted_ind, population)
                population.pop(index)
        return next_gen

    def print_population(self, population):
        for ind in population:
            print(ind.tour)

    """ Calculate metrics of the population """
    def calculate_metrics(self, population, distanceMatrix):
        fitnesses = list()

        for i in range(len(population)):
            fitnesses.append(self.cost(population[i], distanceMatrix))

        min_cost = min(fitnesses)
        index_min_cost = fitnesses.index(min_cost)

        mean_objective = statistics.mean(fitnesses)
        # diversity_estimate = statistics.stdev(fitnesses)

        return mean_objective, population[index_min_cost]

    """ The evolutionary algorithm's main loop """
    def optimize(self, filename):
        # Read distance matrix from file
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        nlen = distanceMatrix.shape[0]

        # ban unconnected cities from population by setting their cost extremely high
        for i in range(nlen-1):
            for j in range(nlen-1):
                if distanceMatrix[i][j] == math.inf:
                    distanceMatrix[i][j] = 1000000
        file.close()

        # initialize parameters and population:
        params = Params(distanceMatrix)
        population = self.init(params, nlen)
        last_best_cost = math.inf
        improvement = True

        it = 0
        while improvement:
            start = time.time()
            it += 1

            # recombination
            offspring = list()
            for i in range(math.ceil(params.amountOfOffspring/2)):
                parent1, index = self.selection(population, params)
                parent2, index = self.selection(population, params)
                # ordered crossover to generate offspring
                self.ordered_crossover(parent1, parent2, offspring, nlen)  # create child
                self.ordered_crossover(parent2, parent1, offspring, nlen)  # second child

            # mutation on the offspring
            for i in range(len(offspring)):
                self.rs_mutation(offspring[i], nlen)

            # mutation seed population
            for i in range(len(population)-1):
                self.rs_mutation(population[i], nlen)

            # combine seed population with offspring into new population
            population.extend(offspring)

            # elimination by crowding or elitism
            #population = self.crowding(population, params)
            population = self.elimination(population, params)  # (l+µ) elimination
            #population = self.elimination(offspring, params) # (l,µ) elimination


            # calculate best individual and mean objective value
            meanObjective, best_ind = self.calculate_metrics(population, distanceMatrix)
            bestObjective = self.cost(best_ind, distanceMatrix)
            bestSolution = best_ind.tour

            itT = time.time() - start
            print(it, ")", f'{itT*1000: 0.1f} ms ',  "mean cost: ", f'{meanObjective:0.2f}', "Lowest/best cost: ",
                  f'{bestObjective:0.2f}', "div.: ", f'{meanObjective-bestObjective:0.2f}')

            if it % 30 == 0:  # check if there is improvement every x iterations
                if last_best_cost < bestObjective + 10:
                    improvement = False
                    print("STOP by no improvement")
                else:
                    last_best_cost = bestObjective

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            # dummy data so that the code runs
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print("STOP by timeout")
                break

        print("best tour", best_ind.tour)
        print("cost best tour", f'{bestObjective: 0.2f}')
        print("execution time", f'{300-timeLeft: 0.2f} sec')
        return 0


# calls optimize function
# todo call optimizer in separate file
class main:
    tsp = r0680462()
    tsp.optimize("tour194.csv")
