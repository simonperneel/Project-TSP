from numpy.core.fromnumeric import mean
import Reporter
import numpy as np
import random
import statistics
import scipy.stats as stats
import math
import time
import matplotlib.pyplot as plt
import pandas as pd


class Individual:
    def __init__(self, tour):
        self.tour = tour
        self.alpha = 0.5        # probability of mutation
        self.prc = 0.99         # probability of recombination
        self.pcw = 0.25         # probability of crowding
        self.cost = 0           # cost of a tour
        self.uptodate = False   # flag if cost of a tour is still up-to-date

class Params:

    # set start parameters
    def __init__(self, distanceMatrix):
        self.popsize = 250  # population size
        self.amountOfOffspring = 150  # amount of trials to generate a child (see prc)
        self.k = 5  # for k-tournament selection
        self.distanceMatrix = distanceMatrix  # matrix with the cost between cities
        self.pheur = 0.5  # percentage of the pop that is initialized with nearest neighbour heuristic


class r0680462:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def cost(self, ind, distanceMatrix):
        tour = ind.tour
        if ind.uptodate:  # no recalculation needed
            cost = ind.cost
        else:
            cost = distanceMatrix[tour[len(tour)-1]][tour[0]]  # cost between first and last city to make circle complete
            for i in range(len(tour) - 1):
                cost += distanceMatrix[tour[i]][tour[i+1]]
            ind.cost = cost
            ind.uptodate = True  # cost is up-to-date
        return cost

    def distance(self, city1, city2, params):
        distance = params.distanceMatrix[city1][city2]
        return distance

    """ amounts of swaps (not minimum) that tours are away from each other, 'distance' """
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

    """ picks one of the 4 mutation operators (weighted) """
    def mutation(self, individual, nlen):
        mutations = [self.swap_mutation, self.insert_mutation,
                     self.rs_mutation, self.rs_mutation, self.scramble_mutation]
        random.choice(mutations)(individual, nlen)

    def swap_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            individual.uptodate = False  # cost of the tour will change
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)

            tour = individual.tour
            tour[i1], tour[i2] = tour[i2], tour[i1]
            individual.tour = tour

    """ randomly selects two cities and inserts one before the other """
    def ordered_mutation(self, individual, nlen):
        if random.random() < individual.alpha:  # alpha % chance of mutation
            individual.uptodate = False
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            tour = individual.tour
            tmp = tour[i1]
            tour = np.delete(tour, i1)
            tour = np.insert(tour, i2, tmp)
            individual.tour = tour

    """ reverses a random selected part of the tour"""
    def rs_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            individual.uptodate = False
            start = random.randint(0, nlen - 1)
            end = start
            while start == end:
                end = random.randint(0, nlen - 1)
            if start > end:
                start, end = end, start
            tour = individual.tour
            tour[start:end+1] = np.flip(tour[start:end+1])  # reverse part of tour

    def insert_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            individual.uptodate = False
            i1 = random.randint(0, nlen - 2)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            if i1 > i2:
                i1, i2 = i2, i1
            tour = individual.tour
            tour = np.insert(tour, i1+1, tour[i2])  # insert value at i2 next to i1
            tour = np.delete(tour, i2+1)
            individual.tour = tour

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

    """ random part of the tour have their positions scrambled"""
    def scramble_mutation(self, individual, nlen):
        if random.random() < individual.alpha:
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            if i1 > i2:
                i1, i2 = i2, i1
            tour = individual.tour
            subtour = tour[i1:i2]
            random.shuffle(subtour)
            tour[i1:i2] = subtour
            individual.tour = tour

    """ picks one of the crossover operators (weighted) """
    def crossover(self, p1, p2, offspring, nlen, params):
        crossovers = [self.order_crossover, self.order_crossover, self.dpx_crossover]
        random.choice(crossovers)(p1, p2, offspring, nlen, params)

    """ order crossover """
    def order_crossover(self, p1, p2, offspring, nlen, params):
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

    """ cycle crossover """
    def cycle_crossover(self, p1, p2, offspring, nlen, params):
        if random.random() < p1.prc:
            p1_tour = p1.tour.tolist()
            p2_tour = p2.tour.tolist()
            p1_tour_copy = p1.tour.tolist()
            p2_tour_copy = p2.tour.tolist()

            child1_tour = np.full(nlen, -1, dtype=int)
            child2_tour = np.full(nlen, -1, dtype=int)
            swap = True
            count = 0
            pos = 0

            while True:
                if count > nlen:
                    break
                for i in range(nlen):
                    if child1_tour[i] == -1:  # empty
                        pos = i
                        break
                if swap:
                    while True:
                        child1_tour[pos] = p1_tour[pos]
                        count += 1
                        pos = p2_tour.index(p1_tour[pos])
                        if p1_tour_copy[pos] == -1:
                            swap = False
                            break
                        p1_tour_copy[pos] = -1
                else:
                    while True:
                        child1_tour[pos] = p2_tour[pos]
                        count += 1
                        pos = p1_tour.index(p2_tour[pos])
                        if p2_tour_copy[pos] == -1:
                            swap = True
                            break
                        p2_tour_copy[pos] = -1

            for i in range(nlen):
                if child1_tour[i] == p1_tour[i]:
                    child2_tour[i] = p2_tour[i]
                else:
                    child2_tour[i] = p1_tour[i]
            for i in range(nlen):
                if child1_tour[i] == -1:
                    if p1_tour_copy == -1:
                        child1_tour[i] = p2_tour[i]
                    else:
                        child1_tour[i] = p1_tour[i]

            child1 = Individual(child1_tour)
            child2 = Individual(child2_tour)
            offspring.append(child1)
            offspring.append(child2)
        else:
            pass  # no recombination happened

    def dpx_crossover(self, p1, p2, offspring, nlen, params):
        tour1 = p1.tour
        tour2 = p2.tour
        # do order crossover for equals tours
        comparison = tour1 == tour2
        if comparison.all():
            self.order_crossover(p1,p2, offspring, nlen, params)

        length = len(tour1)-1

        # convert p1 to a graph structure
        graph_p1 = {tour1[0]: [tour1[1], tour1[length]]}  # edges of the first node
        for i in range(1, length):
            graph_p1[tour1[i]] = [tour1[i+1], tour1[i-1]]
        graph_p1[tour1[length]] = [tour1[0], tour1[length-1]]

        # same for p2
        graph_p2 = {tour2[0]: [tour2[1], tour2[length]]}  # edges of the first node
        for i in range(1,length):
            graph_p2[tour2[i]] = [tour2[i+1], tour2[i-1]]
        graph_p2[tour2[length]] = [tour2[0], tour2[length-1]]

        # create graph with mutual edges of the parents
        child_graph = {}
        for i in range(length+1):
            child_graph[i] = [x for x in graph_p1[i] if x  in graph_p2[i]]

        single_edge_nodes = list()
        zero_or_single_edge_nodes = list()
        double_edge_nodes = list()
        for i in range(len(child_graph)):
            if len(child_graph[i]) == 2:
                double_edge_nodes.append(i)
            elif len(child_graph[i]) ==1:
                zero_or_single_edge_nodes.append(i)
                single_edge_nodes.append(i)
            elif len(child_graph[i]) == 0:
                zero_or_single_edge_nodes.append(i)

        child_tour = np.full(nlen, -1)
        if len(zero_or_single_edge_nodes) > 0:
            start = random.choice(zero_or_single_edge_nodes)
            child_tour[0] = start
            zero_or_single_edge_nodes.remove(child_tour[0])
        else:
            start = random.randint(0, nlen-1)
            child_tour[0] = start

        for i in range(len(child_tour)-1):
            if len(child_graph[child_tour[i]]) != 0:
                child_tour[i+1] = child_graph[child_tour[i]][0]
                child_graph[child_tour[i]].pop(0)  # remove edge between child_tour[i] and child_tour[i+1]
                child_graph[child_tour[i+1]].remove(child_tour[i])  # remove edge between child_tour[i+1] and child_tour[i]
                if child_tour[i+1] in zero_or_single_edge_nodes:
                    zero_or_single_edge_nodes.remove(child_tour[i+1])
            else:
                # node has no edges left
                # zero_or_single_edge_nodes.remove(child_tour[i])

                # calculate the distance to all start nodes
                distances = list()
                for node in zero_or_single_edge_nodes:
                    distances.append(params.distanceMatrix[child_tour[i]][node])

                #todo makes no sense pop => indeces are different in distances => dont match indices in zero_or_single_edge_nodes
                #sort zero_or_single_edge_nodes according to distance and than for loop over the nodes instead of while!
                #print("distances: ", distances)
                #print("zero and ones: ", zero_or_single_edge_nodes)
                zero_or_single_edge_nodes = sorted(zero_or_single_edge_nodes, key=lambda x: distances[zero_or_single_edge_nodes.index(x)])
                #print("sorted zero: ", zero_or_single_edge_nodes)

                for j, node in enumerate(zero_or_single_edge_nodes):
                    if node not in graph_p1[child_tour[i]] and node not in graph_p2[child_tour[i]]:
                        child_tour[i+1] = node
                        zero_or_single_edge_nodes.remove(node)
                        break
                    else:
                        if j == len(zero_or_single_edge_nodes)-1:  # last node in list => add this one
                            child_tour[i+1] = node
                            zero_or_single_edge_nodes.remove(node)

        child = Individual(child_tour)
        offspring.append(child)

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

    """ plot convergence graph """
    def plot(self, filename):
        plt.figure(1)
        plt.autoscale()
        ax = plt.gcf().gca()
        data = pd.read_csv(filename, delimiter=',', header=1)
        data.plot(x='# Iteration', y=' Mean value', kind='line', label='mean value', c='teal', linewidth=2, ax=ax)
        data.plot(x='# Iteration', y=' Best value', kind='line', label='best value', c='firebrick', linewidth=2, ax=ax)
        plt.show()


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
                # order crossover, cycle crossover or dpx crossover to generate offspring
                self.order_crossover(parent1, parent2, offspring, nlen, params)  # create child
                self.order_crossover(parent2, parent1, offspring, nlen, params)  # create second child
                #self.dpx_crossover(parent1, parent2, offspring, nlen, params)
                #self.dpx_crossover(parent2, parent1, offspring, nlen, params)
                #self.cycle_crossover(parent1, parent2, offspring, nlen, params)
                #self.crossover(parent1, parent2, offspring, nlen, params)  # picks random one of the crossover operators
                #self.crossover(parent2, parent1, offspring, nlen, params)

            # mutation on the offspring
            for i in range(len(offspring)):
                self.mutation(offspring[i], nlen)

            # mutation seed population
            for i in range(1, (len(population)-1)):  # 0/1 to (not) mutate best individual
                self.mutation(population[i], nlen)

            # combine seed population with offspring into new population
            population.extend(offspring)

            # elimination by crowding or elitism
            #population = self.crowding(population, params)
            population = self.elimination(population, params)  # (l+µ) elimination
            #population = self.elimination(offspring, params)  # (l,µ) elimination


            # calculate best individual and mean objective value
            meanObjective, best_ind = self.calculate_metrics(population, distanceMatrix)
            bestObjective = self.cost(best_ind, distanceMatrix)
            bestSolution = best_ind.tour

            # dynamic parameters, don't improve the solution much
           # for ind in population:
           #     ind.alpha = (1-((meanObjective-bestObjective)/bestObjective))**0.4
            #print("alpha ", population[0].alpha)

            itT = time.time() - start
            print(it, ")", f'{itT*1000: 0.1f} ms ',  "mean cost: ", f'{meanObjective:0.2f}', "Lowest/best cost: ",
                  f'{bestObjective:0.2f}', "div.: ", f'{meanObjective-bestObjective:0.2f}')

            #if it % 50 == 0:  # check if there is improvement every x iterations
             #   if last_best_cost <= bestObjective:
              #      improvement = False
               #     print("STOP by no improvement")
              #  else:
               #     last_best_cost = bestObjective

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
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
    tsp.optimize("tour929.csv")
    tsp.plot("r0680462.csv")
