# Project-TSP
## Problem statement
The traveling salesperson problem consists of minimizing the cost of a cycle that visits all vertices in a
weighted, directed graph. The cost of a cycle is defined as the sum of the weights of all directed edges constituting the cycle. For example, the cycle defined by the red edges in the figure below has a cost of 10:
<br/>
<br/>
<p align="center">
<img src="img\graph.png" width=400 align=center>
</p>  
<br/>
<br/> 

Hence, the optimization problem involves visiting all vertices, while keeping the cost of the tour to a minimum. The goal of this project is to solve this optimization problem using an **evolutionary algorithm**. A local search operator is added to the framework to accelerate the search:


<br/>
<br/>
<p align="center">
  <img src="img\algorithm.png" width="413" />
  <img src="img\localsearch.png" width="400" /> 
</p>

## Initialization
The multiset of candidate solutions is called the "population". A candidate solution is a cycle that visits all vertices in the graph, or a "tour" that visits every "city". A tour appears as a permutation of integers in the code and the objective value of a tour is its cost. The initial population should have a good coverage of the domain. Therefore, random initialization is mostly used to initialize the population. A small part of the initial population is initialized with a local search operator (nearest neigbour), to speed up the search.

## Selection operators
The aim of the selection operator is to select, with replacement, candidate solutions with a good objective value (here: a tour with a low cost). Pairs of these selected candidate solutions are called 'parents'. They serve as input to the recombination operator, resulting in a new candidate solution, a 'child'. **k-tournament** selection is used in this project. The implementation of all operators of the algorithm is described in the [code](/code) folder. 

## Variation operators
The goal of variation operators is to generate new candidate solutions, "offspring", from the parents. The crossover operator amplifies promising features of the selected parents, while the mutation operator introduce randomness by perturbing its input candidate solution
### Crossover operator
The used crossover operator in the project are **order crossover** and **DPX crossover**. Cycle crossover is implemented but not used. 
### Mutation operator
The used mutation operators for the tours are operators for permutations. These include **swap mutation**, **reverse sequence mutation**, **scramble mutation** and **insert mutation** 

## Elimination scheme
Elimination is used to prevent the population from growing indefinitively. The goal is to prune the population to the promising areas of the search space. On the one hand, the elimination scheme has to promote exploitation (by removing candidate solutions with a bad objective value) and on the other, it should maintain diversity in the population so that multiple good candidate individuals may be found. In this project, **elitism** is used in combination with **crowding**.


## Reference
This documentation is based on the slides of the course *Genetic algorithms and Ev  olutionary computing*, taught by Prof. dr. N. Vannieuwenhoven, KU Leuven, 2020-2021.

