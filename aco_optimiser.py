import numpy as np

# Define the Ant class to represent an individual ant in the ACO algorithm
class Ant:
    def __init__(self, buy_dnf=None, sell_dnf=None):
        # Initialize the ant with optional buy and sell DNFs
        self.buy_dnf = buy_dnf
        self.sell_dnf = sell_dnf

    def set_buy_dnf(self, buy_dnf):
        # Set the buy DNF for the ant
        self.buy_dnf = buy_dnf

    def set_sell_dnf(self, sell_dnf):
        # Set the sell DNF for the ant
        self.sell_dnf = sell_dnf


# Define the ACOOptimiser class to run the ACO algorithm
class ACOOptimiser:
    def __init__(self, cost_function, keys):
        # Initialize the ACOOptimiser with a cost function and literal keys

        # Set the parameters for the ACO algorithm
        self.cost_function = cost_function
        self.literals = list(keys)
        self.num_ants = 10
        self.num_iterations = 1
        self.alpha = None
        self.beta = None
        self.evaporation_rate = None
        self.max_clauses = 10

        # Set the parameters for checking convergence
        self.max_no_improvement_iterations = 50
        self.current_no_improvement_iterations = 0
        self.best_fitness = float('inf')

        # Create a list of ant objects
        self.ants = [Ant() for _ in range(self.num_ants)]

        # Create a list of literals for each key, along with their negations
        self.literals = [(literal, val) for literal in self.literals for val in [True, False]]

        # Initialize the pheromone values for buy and sell literals
        self.buy_pheremones = [1 for _ in range(2 * len(self.literals))]
        self.sell_pheremones = [1 for _ in range(2 * len(self.literals))]
        self.normalise_pheremones()

    def normalise_pheremones(self):
        self.buy_pheremones = [x/sum(self.buy_pheremones) for x in self.buy_pheremones]
        self.sell_pheremones = [x/sum(self.sell_pheremones) for x in self.sell_pheremones]

    def initialise_ants(self):
        """
            This function creates an initial population of ants.
            Each ant should have an initial solution (DNF formula) for buy and sell, 
            randomly generated based on the given pheromone probabilities. 
            Remember, the DNF formula is represented by a list of sets, and each set value is a tuple.
              
              [{(indicator1, False), (indicator2, True)}, {indicator3, True}]

            This example above represents the DNF formula: (indicator1 OR indicator2) AND indicator3
            This can be achieved by sampling literals using the probabilities 
            in the buy_pheromones and sell_pheromones.
        """
        # Create a list of ants with random DNF formulas
        pass

    def construct_solution(self, ant):
        """
        This function updates the solution of the given ant by applying local search operators
        (e.g., adding/removing/changing literals or clauses) based on the pheromone probabilities
        The new solution should respect the max_clauses constraint.
        """
        # Modify the ant's DNF formula using probabilities derived from the pheromone matrix
        pass

    def evaluate_solution(self, ant):
        """
        Evaluate a DNF formula using the provided cost function.
        """
        if ant.buy_dnf is None:
            return None
        return self.cost_function(ant.buy_dnf, ant.sell_dnf)


    def update_pheromone(self, solutions_fitness):
        """
            This function updates the pheromone levels of the buy and sell matrices based on the fitness values of the ants' solutions.
            The evaporation_rate, alpha, and beta parameters should be used to control the pheromone update process.
            Normalise the matrices after updating the pheromone levels.
        """
        # Update the pheromone levels and apply evaporation
        pass

    def extract_best_solution(self, solutions_fitness):
        """
        This function finds and returns the best solution (buy and sell DNFs) among all ants
        based on the fitness values in solutions_fitness.
        You can use the argmin() function from NumPy to find the index of the minimum fitness value
        and then extract the corresponding buy and sell DNFs from the ant.
        """

        # Find the index of the ant with the best (lowest) fitness solution
        best_ant_index = np.argmin(solutions_fitness)

        # Extract the best buy and sell DNF formulas from the ant with the best solution
        best_buy_dnf_formula = self.ants[best_ant_index].buy_dnf
        best_sell_dnf_formula = self.ants[best_ant_index].sell_dnf

        return best_buy_dnf_formula, best_sell_dnf_formula

        # uncomment this below and run run_bot.py to see the output of the algorithm.
        #return [{(self.literals[1], True), (self.literals[2], False)}], [{(self.literals[0], True), (self.literals[2], False)}]

    def check_convergence(self):
        """
        This function checks if the algorithm has converged based on some stopping criteria
        (e.g., a max cost value or a maximum number of iterations without improvement).
        If the stopping criteria are met, the function returns True; otherwise, it returns False.
        """
        # Find the best fitness value among all ants
        current_best_fitness = min([self.evaluate_solution(ant) for ant in self.ants])

        # Check if the best fitness value has improved compared to the previous iteration
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.current_no_improvement_iterations = 0
        else:
            self.current_no_improvement_iterations += 1

        # Check if the maximum number of iterations without improvement has been reached
        if self.current_no_improvement_iterations >= self.max_no_improvement_iterations:
            return True
        else:
            return False

    def aco_algorithm(self):
        """
        Run the ACO algorithm for the given number of iterations and return the best DNF formula.
        Uses num_literals, num_ants, num_iterations, alpha, beta, evaporation_rate, max_clauses
        """
        
        
        # Initialise ants
        self.initialise_ants()

        # Main loop
        
        for iteration in range(self.num_iterations):
            solutions_fitness = []
            # Construct solutions
            for ant in self.ants:
                if iteration > 0:
                    self.construct_solution(ant)
                solutions_fitness.append(self.evaluate_solution(ant))
            # Evaluate solutions, need to pass in the trading bot instance?
            

            # Update pheromone levels
            self.update_pheromone(solutions_fitness)

            if self.check_convergence():
                break

        # Extract the best solution
        best_buy_dnf_formula, best_sell_dnf_formula = self.extract_best_solution(solutions_fitness)

        return (best_buy_dnf_formula, best_sell_dnf_formula)


