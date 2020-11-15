from numpy.core.fromnumeric import mean
import Reporter
import numpy as np
import random
import statistics
import time


class Individual:
    def __init__(self, tour):
        self.tour = tour
        self.alpha = 0.05

class Params:

    # set start parameters
    def __init__(self, distanceMatrix):
        self.popsize = 200  # population size
        self.amountOfOffspring = 100  # amount individuals in the offspring
        self.alpha = 0.05  # probability of mutation
        self.k = 15  # for k-tournament selection
        self.distanceMatrix = distanceMatrix  # matrix with the cost between cities

class r0680462:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def cost(self, ind, distanceMatrix):
        # calculate cost
        tour = ind.tour
        cost = 0
        for i in range(len(tour) - 1):
            cost += distanceMatrix[tour[i]][tour[i + 1]]
        return cost

    def init(self, params, nlen):
        population = []
        for i in range(params.popsize):
            tour = np.arange(nlen)
            np.random.shuffle(tour)
            ind = Individual(tour)
            population.append(ind)
        return population

    def selection(self, params, population):
        inds = []
        for i in range(params.k):
            r = random.randint(0, len(population) - 1)
            inds.append(r)
        all_costs = []
        for i in range(params.k):
            all_costs.append(
                self.cost(population[inds[i]], params.distanceMatrix))
        index = all_costs.index(min(all_costs))
        return population[inds[index]]

    def swap_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while (i1 == i2):
                i2 = random.randint(0, nlen - 1)

            tour = individual.tour
            tour[i1], tour[i2] = tour[i2], tour[i1]
            individual.tour = tour

    # randomly selects two cities and inserts one before the other
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

    # only keep the best individuals in the population, eliminate others
    def elimination(self, population, params):
        population = sorted(population, key=lambda x: self.cost(x, params.distanceMatrix))
        return population[:params.popsize]

    # crossover function
    def ordered_crossover(self, p1, p2, nlen):
        # start index for subset that will be transferred to the child
        i1 = random.randint(0, nlen - 2)
        i2 = random.randint(i1 + 1, nlen - 1)  # end index of subset

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
            if (child_tour[i] == -1):  # empty spot
                child_tour[i] = p2_tour[j]
                j += 1

        # create child
        child = Individual(child_tour)
        return child

    def print_population(self, population):
        for ind in population:
            print(ind.tour)

    # Calculate metrics of the population
    def calculate_metrics(self, population, distanceMatrix):
        fitnesses = list()

        for i in range(len(population)):
            fitnesses.append(self.cost(population[i], distanceMatrix))

        min_cost = min(fitnesses)
        index_min_cost = fitnesses.index(min_cost)

        mean_objective = statistics.mean(fitnesses)

        return mean_objective, population[index_min_cost]


    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # set parameters:
        params = Params(distanceMatrix)
        maxit = 1000
        nlen = distanceMatrix.shape[0]

        # init population
        population = self.init(params, nlen)
        last_best_cost = 10000000
        improvement = True

        it = 0
        while (it < maxit) & improvement:
            it += 1

            # recombination
            offspring = list()
            for i in range(params.amountOfOffspring):
                parent1 = self.selection(params, population)
                parent2 = self.selection(params, population)
                # ordered crossover for offspring
                offspring.append(self.ordered_crossover(parent1, parent2, nlen))  # first child
                offspring.append(self.ordered_crossover(parent2, parent1, nlen))  # second child
                # swap mutation on the offspring
                self.swap_mutation(offspring[i], nlen)

            # mutation seed population
            for i in range(params.popsize-1):
                self.swap_mutation(population[i], nlen)

            # combine seed population with offspring into new population
            population.extend(offspring)

            # elimination
            population = self.elimination(population, params)
            #self.print_population(population)
            # calculate best individual and mean objective value
            meanObjective, best_ind = self.calculate_metrics(population, distanceMatrix)
            bestObjective = self.cost(best_ind, distanceMatrix)
            bestSolution = best_ind.tour

            print(it, ")", "mean cost: ", meanObjective, "Lowest/best cost: ", bestObjective)


            if it % 20 == 0:  # check if there is improvement every 20 iterations
                if last_best_cost < bestObjective + 50:
                    improvement = False
                    print("STOP by no improvement")
                else:
                    last_best_cost = bestObjective

            if it > 50:
                for ind in population:
                    ind.alpha = 0.5


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

        return 0


# calls optimize function
class main:
    tsp = r0680462()
    tsp.optimize("tour29.csv")
