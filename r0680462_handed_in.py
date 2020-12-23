import Reporter
import numpy as np
import random
import statistics
import scipy.stats as stats
import math
import time
import matplotlib.pyplot as plt
import pandas as pd


class Params:

    # set start parameters
    def __init__(self, dM):
        self.popsize = 250              # population size
        self.amountOfOffspring = 150    # amount of trials to generate a child (see prc)
        self.k = 5                      # for k-tournament selection
        self.distanceMatrix = dM        # matrix with the cost between cities
        self.pheur = 0.25               # % of the pop that is initialized with nearest neighbour heuristic
        self.minalpha = 0.3             # minimum value for alpha
        self.maxalpha = 0.5             # maximum value for alpha
        self.alpha = self.minalpha      # probability of mutation
        self.pcw = 0.15                 # probability of crowding


class Individual:
    def __init__(self, tour):
        self.tour = tour                # tour of cities
        self.cost = 0                   # cost of an individual
        self.prc = 0.99                 # probability of recombination
        self.edgeset = {}               # set of edges from the tour
        self.cost_uptodate = False      # flag if cost of a tour is still up-to-date
        self.edges_uptodate = False     # flag to check if edge set of a tour is still up-to-date


class r0680462:

    """ initializes the reporter and starts the counter """
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    """ calculate the cost of a tour, used as fitness function """
    def cost(self, ind, distanceMatrix):
        tour = ind.tour
        if ind.cost_uptodate:  # no recalculation needed
            cost = ind.cost
        else:
            cost = distanceMatrix[tour[len(tour)-1]][tour[0]]  # cost between first and last city to make circle complete
            for i in range(len(tour) - 1):
                cost += distanceMatrix[tour[i]][tour[i+1]]
            ind.cost = cost
            ind.cost_uptodate = True  # cost is updated
        return cost

    """ calculate the distance between two cities """
    def distance(self, city1, city2, params):
        distance = params.distanceMatrix[city1][city2]
        return distance

    """ calculate the amount of swaps (not minimum) that tours are away from each other, used
        as distance between two tours """
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

    """ calculate the amount of different cities between two tours """
    def hamming_distance(self, ind1, ind2):
        distance = 0
        for i in range(len(ind1.tour)):
            if ind1.tour[i] != ind2.tour[i]:
                distance += 1
        return distance

    """ calculate kendall tau distance, a correlation measure for lists
        value close to 1: strong agreement, close to 0: strong disagreement """
    def kendalltau_distance(self, ind1, ind2):
        distance, pvalue = stats.kendalltau(ind1.tour, ind2.tour)
        return distance

    """ calculate amount of shared edges between an individuals tour and
        returns the amount of different edges as distance"""
    def different_edges_distance(self, ind1, ind2):
        tour1 = ind1.tour
        tour2 = ind2.tour
        length = len(tour1)

        if ind1.edges_uptodate:
            edges1 = ind1.edgeset
        else:  # recalculate edge set of first tour
            edge = tour1[0], tour1[-1]
            edges1 = {edge}
            for i in range(1, len(tour1)):
                edge = tour1[i], tour1[i-1]
                edges1.add(edge)

        # same for second tour
        if ind2.edges_uptodate:
            edges2 = ind2.edgeset
        else:
            edge = tour2[0], tour2[-1]
            edges2 = {edge}
            for i in range(1, len(tour1)):
                edge = tour2[i], tour2[i-1]
                edges2.add(edge)

        return length - len(edges1.intersection(edges2))  # return amount of different edges

    """ returns closest individual of a sample/all individuals from the population to a given individual """
    def closestIndividual(self, ind, population):
        all_distances = []
        #print("pop length", len(population))
        #for i in range(len(population)):
         #   all_distances.append(self.different_edges_distance(ind, population[i]))
        #index = all_distances.index(min(all_distances))
        # closest to sampled individuals
        inds = []
        for i in range(5):  # sample 5 individuals
            inds.append(random.randint(0, len(population) - 1))  # fill list with indices of the sampled inds
            all_distances.append(self.different_edges_distance(ind, population[inds[i]]))  # calculate distances
        index = all_distances.index(min(all_distances))  # index of closest individual
        return population[inds[index]], inds[index]

    """ initializes the population with a % of heuristic individuals """
    def init(self, params, nlen):
        population = []
        # random initialization
        for i in range(int(params.popsize*(1-params.pheur))):
            tour = np.random.permutation(nlen)
            ind = Individual(tour)
            population.append(ind)
        # initialize % of pop with heuristic good individuals
        for i in range(int(params.popsize*params.pheur)):
            tour = self.init_nn(params, nlen)
            ind = Individual(tour)
            population.append(ind)
        return population

    """ initializes a tour with nearest neighbours heuristic """
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
        return tour

    """ find the city that is nearest to city A """
    def nearestneighbor(self, A, cities, params):
        return min(cities, key=lambda c: self.distance(c, A, params))

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

    """ picks one of the 4 mutation operators (weighted) """
    def mutation(self, individual, nlen, params):
        mutations = [self.swap_mutation,
                     self.insert_mutation,
                     self.rs_mutation, self.rs_mutation, self.rs_mutation, self.rs_mutation,
                     self.scramble_mutation, self.scramble_mutation]
        random.choice(mutations)(individual, nlen, params)

    """ randomly swaps two cities of a tour """
    def swap_mutation(self, individual, nlen, params):
        if random.random() < params.alpha:
            individual.cost_uptodate = False  # cost and edge set of the tour will change
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            tour = individual.tour
            tour[i1], tour[i2] = tour[i2], tour[i1]
            individual.tour = tour

    """ reverses a random selected part of the tour"""
    def rs_mutation(self, ind, nlen, params):
        if random.random() < params.alpha   :
            ind.cost_uptodate = ind.edges_uptodate = False
            start = random.randint(0, nlen - 1)
            end = start
            while start == end:
                end = random.randint(0, nlen - 1)
            if start > end:
                start, end = end, start
            tour = ind.tour
            tour[start:end+1] = np.flip(tour[start:end+1])  # reverse part of tour

    """ inserts a random picked city in a tour """
    def insert_mutation(self, ind, nlen, params):
        if random.random() < params.alpha:
            ind.cost_uptodate = ind.edges_uptodate = False
            i1 = random.randint(0, nlen - 2)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            if i1 > i2:
                i1, i2 = i2, i1
            tour = ind.tour
            tour = np.insert(tour, i1+1, tour[i2])  # insert value at i2 next to i1
            tour = np.delete(tour, i2+1)
            ind.tour = tour

    """ random part of the tour have their positions scrambled"""
    def scramble_mutation(self, ind, nlen, params):
        if random.random() < params.alpha:
            ind.cost_uptodate = ind.edges_uptodate = False
            i1 = random.randint(0, nlen - 1)
            i2 = i1
            while i1 == i2:
                i2 = random.randint(0, nlen - 1)
            if i1 > i2:
                i1, i2 = i2, i1
            tour = ind.tour
            subtour = tour[i1:i2]
            random.shuffle(subtour)
            tour[i1:i2] = subtour
            ind.tour = tour

    """ picks one of the crossover operators (weighted) """
    def crossover(self, p1, p2, offspring, nlen, params):
        crossovers = [self.order_crossover, self.order_crossover, self.order_crossover, self.order_crossover,
                      self.order_crossover, self.order_crossover, self.order_crossover, self.order_crossover,
                      self.order_crossover, self.dpx_crossover]
        random.choice(crossovers)(p1, p2, offspring, nlen, params)

    """ order crossover """
    def order_crossover(self, p1, p2, offspring, nlen, params=None):
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
    def cycle_crossover(self, p1, p2, offspring, nlen, params=None):
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

    """ distance preserving crossover"""
    def dpx_crossover(self, p1, p2, offspring, nlen, params):
        tour1 = p1.tour
        tour2 = p2.tour
        # no dpx crossover but order crossover for equals tours
        comparison = tour1 == tour2
        if comparison.all():
            self.order_crossover(p1, p2, offspring, nlen)

        # convert p1 to a graph structure
        graph_p1 = {tour1[0]: [tour1[1], tour1[-1]]}  # edges of the first node
        for i in range(1, len(tour1) - 1):
            graph_p1[tour1[i]] = [tour1[i+1], tour1[i-1]]
        graph_p1[tour1[-1]] = [tour1[0], tour1[-2]]

        # same for p2
        graph_p2 = {tour2[0]: [tour2[1], tour2[-1]]}  # edges of the first node
        for i in range(1, len(tour2) - 1):
            graph_p2[tour2[i]] = [tour2[i+1], tour2[i-1]]
        graph_p2[tour2[-1]] = [tour2[0], tour2[-2]]

        # create graph with mutual edges of the parents
        child_graph = {}
        for i in range(len(tour1)):
            child_graph[i] = [x for x in graph_p1[i] if x in graph_p2[i]]

        node_se = list()  # list for the node with single edges
        node_z_or_se = list()  # list for the zero edge nodes and single edge nodes
        node_de = list()  # list for the nodes with double edges
        for i in range(len(child_graph)):
            if len(child_graph[i]) == 2:
                node_de.append(i)
            elif len(child_graph[i]) == 1:
                node_se.append(i)
                node_z_or_se.append(i)
            elif len(child_graph[i]) == 0:
                node_z_or_se.append(i)

        child_tour = np.full(nlen, -1)  # create empty array
        if len(node_z_or_se) > 0:
            start = random.choice(node_z_or_se)
            child_tour[0] = start
            node_z_or_se.remove(child_tour[0])
        else:
            start = random.randint(0, nlen-1)
            child_tour[0] = start

        for i in range(len(child_tour) - 1):
            if len(child_graph[child_tour[i]]) != 0:
                child_tour[i+1] = child_graph[child_tour[i]][0]
                child_graph[child_tour[i]].pop(0)  # remove edge between child_tour[i] and child_tour[i+1]
                child_graph[child_tour[i+1]].remove(child_tour[i])  # remove edge between child_tour[i+1] and child_tour[i]
                if child_tour[i+1] in node_z_or_se:
                    node_z_or_se.remove(child_tour[i+1])
            else:
                # no edges left
                # sort all distances from node to start nodes
                distances = list()
                for node in node_z_or_se:
                    distances.append(params.distanceMatrix[child_tour[i]][node])

                node_z_or_se = sorted(node_z_or_se, key=lambda x: distances[node_z_or_se.index(x)])

                for j, node in enumerate(node_z_or_se):
                    if node not in graph_p1[child_tour[i]] and node not in graph_p2[child_tour[i]]:
                        child_tour[i+1] = node
                        node_z_or_se.remove(node)
                        break
                    else:
                        if j == len(node_z_or_se)-1:  # add last node in list to child
                            child_tour[i+1] = node
                            node_z_or_se.remove(node)

        # make an individual of the child tour and append to the offspring
        child = Individual(child_tour)
        offspring.append(child)

    """ only keep the best individuals in the population, eliminate others """
    def elimination(self, population, params):
        population = sorted(population, key=lambda x: self.cost(x, params.distanceMatrix))
        return population[:params.popsize]

    """ crowding with k-tournament promotion """
    def crowdingK(self, population, params):
        next_gen = []
        # promote ind to next generation by k-tournament
        for i in range(params.popsize):
            promoted_ind, index = self.selection(population, params)
            next_gen.append(promoted_ind)  # add to next generation
            population.pop(index)  # remove from seed population
            if random.random() < params.pcw:  # probability of crowding close individual
                closestInd, index = self.closestIndividual(promoted_ind, population)  # find closest individual from promoted one
                population.pop(index)

        return next_gen

    """ crowding with elitism promotion, crowding only starts after 30 iterations
        ( to ban the tours with unconnected cities first) """
    def crowding(self, population, params):
        next_gen = []
        # promote ind to next generation by elitism
        population = sorted(population, key=lambda x: self.cost(x, params.distanceMatrix))  # rank population based on fitness
        for i in range(params.popsize):
            promoted_ind = population[i]
            next_gen.append(promoted_ind)  # copy to next generation
            #population.pop(i)  # remove promoted ind from population, not needed here
            if random.random() < params.pcw:
                closestInd, index = self.closestIndividual(promoted_ind, population[i:len(population)])
                population.pop(index)
        return next_gen

    """ Calculate metrics of the population """
    def calculate_metrics(self, population, distanceMatrix):
        fitnesses = list()

        for i in range(len(population)):
            fitnesses.append(self.cost(population[i], distanceMatrix))

        min_cost = min(fitnesses)
        index_min_cost = fitnesses.index(min_cost)

        mean_objective = statistics.mean(fitnesses)
        #diversity_estimate = statistics.stdev(fitnesses)

        return mean_objective, population[index_min_cost]

    """ plots a convergence graph """
    def plot(self, filename):
        plt.figure(1)
        plt.autoscale()
        ax = plt.gcf().gca()
        data = pd.read_csv(filename, delimiter=',', header=1)
        data.plot(x=' Elapsed time', y=' Mean value', kind='line', label='mean value', c='teal', linewidth=2, ax=ax)
        data.plot(x=' Elapsed time', y=' Best value', kind='line', label='best value', c='firebrick', linewidth=2, ax=ax)
        plt.xlabel('Elapsed time [s]')
        plt.ylabel('Cost')
        plt.grid(alpha=0.6, linewidth=0.3)
        plt.show()

    """ The evolutionary algorithm's main loop """
    def optimize(self, filename):
        # Read distance matrix from file
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # initialize parameters
        nlen = distanceMatrix.shape[0]
        params = Params(distanceMatrix)

        # ban unconnected cities from population by setting their cost extremely high
        for i in range(nlen):
            for j in range(nlen):
                if distanceMatrix[i][j] == math.inf:
                    distanceMatrix[i][j] = 999999
                    params.pcw = 0.05  # small crowding to avoid impossible tours reach next generation

        # initialize population
        population = self.init(params, nlen)

        # start loop
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
                #self.order_crossover(parent1, parent2, offspring, nlen)  # create child
                #self.order_crossover(parent2, parent1, offspring, nlen)  # create second child
                #self.dpx_crossover(parent1, parent2, offspring, nlen, params)
                #self.dpx_crossover(parent2, parent1, offspring, nlen, params)
                #self.cycle_crossover(parent1, parent2, offspring, nlen, params)  # generates two childs
                self.crossover(parent1, parent2, offspring, nlen, params)  # picks random one of the crossover operators (weighted)
                self.crossover(parent2, parent1, offspring, nlen, params)  # second child

            # mutation on the offspring
            for i in range(0, len(offspring)):  # 0/1 to (not) mutate best individual from offspring
                self.mutation(offspring[i], nlen, params)

            # mutation seed population
            for i in range(1, (len(population)-1)):  # 1 to not mutate the best three individual of the population
                self.mutation(population[i], nlen, params)


            # combine seed population with offspring into new population
            population.extend(offspring)

            # elimination by crowding or elitism
            population = self.crowding(population, params)  # (l+µ) elimination with crowding
            #population = self.elimination(population, params)  # (l+µ) elimination
            #population = self.elimination(offspring, params)  # (l,µ) elimination


            # calculate best individual and mean objective value
            meanObjective, best_ind = self.calculate_metrics(population, distanceMatrix)
            bestObjective = self.cost(best_ind, distanceMatrix)
            bestSolution = best_ind.tour

            itT = time.time() - start
            print(it, ")", f'{itT*1000: 0.1f} ms ',  "mean cost: ", f'{meanObjective:0.2f}', "Lowest/best cost: ",
                  f'{bestObjective:0.2f}', "div.: ", f'{meanObjective-bestObjective:0.2f}')

            # stop criterion
            """if it % 50 == 0:  # check if there is improvement every x iterations
                if last_best_cost <= bestObjective:
                    improvement = False
                    print("STOP by no improvement")
                else:
                    last_best_cost = bestObjective"""

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                print("STOP by timeout")
                break

            # dynamically increase alpha every 2s: [minalpha < alpha < maxalpha]
            elapsedTime = 300 - timeLeft
            if int(elapsedTime) % 2 == 0:
                params.alpha = (params.maxalpha - params.minalpha)/300 * elapsedTime + params.minalpha

        # output some values at the end
        print("best tour", best_ind.tour)
        print("cost best tour", f'{bestObjective: 0.2f}')
        print("execution time", f'{300-timeLeft: 0.2f} sec')

        return meanObjective, bestObjective
