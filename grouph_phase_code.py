from numpy.core.fromnumeric import mean
import Reporter
import numpy as np
import random
import statistics

class Individual:
	def __init__(self,tour):
		self.tour = tour
		self.alpha = 0.05

# Modify the class name to match your student number.
class r0664732:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	def cost(self,ind,distanceMatrix):
		#calculate cost
		tour = ind.tour
		cost = 0
		for i in range(len(tour)-1):
			cost += distanceMatrix[tour[i]][tour[i+1]]
		return cost

	def init(self, nlen, popsize):
		population = []
		for i in range(popsize):
			tour = np.arange(nlen)
			np.random.shuffle(tour)
			ind = Individual(tour)
			population.append(ind)
		return population

	def selection(self, k, population,distanceMatrix):
		inds=[]
		for i in range(k):
			r = random.randint(0,len(population)-1)
			inds.append(r)
		all_costs = []
		for i in range(k):
			all_costs.append(self.cost(population[inds[i]],distanceMatrix))
		index = all_costs.index(min(all_costs))
		return population[inds[index]]

	def swap_mutation(self,individual,nlen):
		if(random.random()<individual.alpha):
			i1 = random.randint(0,nlen-1)
			i2 = i1
			while(i1 == i2):
				i2 = random.randint(0,nlen-1)
			
			tour = individual.tour
			tour[i1],tour[i2] = tour[i2],tour[i1]
			individual.tour = tour

	def ordered_mutation(self,individual,nlen):#randomly selects two cities and inserts one before the other
		if(random.random()<individual.alpha):
			i1 = random.randint(0,nlen-1)
			i2 = i1
			while(i1 == i2):
				i2 = random.randint(0,nlen-1)
			tour = individual.tour
			tmp = tour[i1]
			tour = np.delete(tour,i1)
			tour = np.insert(tour,i2,tmp)
			individual.tour = tour

	def elimination(self,population,popsize,distanceMatrix):
		population = sorted(population,key=lambda x:self.cost(x,distanceMatrix))
		return population[:popsize]

	def ordered_crossover(self,p1,p2,nlen):
		i1 = random.randint(0,nlen-2) #start index for subset that will be transferd to the child
		i2 = random.randint(i1+1,nlen-1) #end index of subset	

		child_tour = np.full(nlen,-1, dtype = int)
		for i in range(i1,i2):# copy subset of p1 to child
			child_tour[i] = p1.tour[i] 
		
		#delete values that are already in child from p2
		p2_tour = p2.tour
		values_to_delete = p1.tour[i1:i2]	#values to delete from p2
		for i in range(len(values_to_delete)):
			p2_tour = p2_tour[p2_tour != values_to_delete[i]] # removes element values_to_delete[i] from p2
	
		#insert remaining values of p2 into child
		j=0
		for i in range(len(child_tour)):
			if(child_tour[i]==-1):
				child_tour[i] = p2_tour[j]
				j+=1

		#create child
		child = Individual(child_tour)
		return child

	def calculate_metrics(self, population,distanceMatrix):
		fitnesses = list()
		for i in range(len(population)):
			fitnesses.append(self.cost(population[i],distanceMatrix))

		min_cost = min(fitnesses)
		index_min_cost = fitnesses.index(min_cost)
		
		mean_objective = statistics.mean(fitnesses)
		
		return mean_objective,population[index_min_cost]


	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		#params:
		popsize = 250
		nlen = distanceMatrix.shape[0]
		k = 10
		maxit = 100
		amountOfOffspring = 75

		#inint population
		population = self.init(nlen,popsize)
		last_best_cost =1000000040
		no_improvement = False
		# Your code here.31
		it = 0
		while((it<maxit) & (no_improvement == False)):
			it +=1
			# Your code here.

			# Recombination
			offspring = list()
			for i in range(amountOfOffspring):
				parent1 = self.selection(k,population,distanceMatrix)
				parent2 = self.selection(k,population,distanceMatrix)
				offspring.append(self.ordered_crossover(parent1,parent2,nlen))
				self.swap_mutation(offspring[i],nlen) #ordered mutation for offspring

			#mutation seed population
			for i in range(popsize):
				self.swap_mutation(population[i],nlen)

			#combine seed population with offspring into new population
			population.extend(offspring)

			#elimination
			population = self.elimination(population,popsize,distanceMatrix)

			#calculate best individual and mean objective value
			meanObjective, best_ind = self.calculate_metrics(population,distanceMatrix)
			bestObjective = self.cost(best_ind,distanceMatrix)
			bestSolution = best_ind.tour
			
			print(it,")" , "mean cost: ", meanObjective, "Lowest/best cost: ", bestObjective)

			if(it%20 == 0):# check if there is improvement every 10 itterations
				if last_best_cost < bestObjective + 50:
					no_improvement = True
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
				break

		print("time = ", (300-timeLeft))


		# Your code here.
		return 0


#calls optimize function
class main:
	tsp = r0664732()
	tsp.optimize("tour29.csv")
