import numpy as np
import copy
import random 

# Define the Ant class to represent an individual ant in the ACO algorithm
class Ant:
    def __init__(self, buy_dnf=None, sell_dnf=None):
        # Initialize the ant with optional buy and sell DNFs
        self.buy_dnf = buy_dnf
        self.sell_dnf = sell_dnf



# Define the ACOOptimiser class to run the ACO algorithm
class ACOOptimiser:
    def __init__(self, cost_function, keys):
        # Initialize the ACOOptimiser with a cost function and literal keys

        # Set the parameters for the ACO algorithm
        self.cost_function = cost_function
        self.literals = list(keys)

        self.num_iterations = 1
        self.alpha = None
        self.beta = None
        self.evaporation_rate = None
        
        self.average_clauses = 2
        self.average_literals = 2
        self.num_ants = 500


        # All our ants for the ACO algorithm
        self.ants = [Ant() for _ in range(self.num_ants)]

        # A list of our best-found ants while optimising.
        self.best_ants = []

        # Set the parameters for checking convergence
        # Current unimproved iteration set to 0 if a best ant is added to list.
        self.max_unimproved_iterations = 50
        self.current_unimproved_iteration = 0


        # A list of literals for each key, along with their negations
        self.literals = [(literal, val) for literal in self.literals for val in [True, False]]

        # Initialize the pheromone values for buy and sell literals
        self.buy_pheremones = [1 for _ in range(len(self.literals))]
        self.sell_pheremones = [1 for _ in range(len(self.literals))]
        self.normalise_pheremones()

    def normalise_pheremones(self):
        self.buy_pheremones = [x/sum(self.buy_pheremones) for x in self.buy_pheremones]
        self.sell_pheremones = [x/sum(self.sell_pheremones) for x in self.sell_pheremones]

    def initialise_ants(self):
        """
            This function creates an initial population of ants.
            Each ant should have an initial solution (DNF formula) for buy and sell, 
            randomly generated based on the given pheromone probabilities. 
            Remember, the DNF formula is represented by a list of sets, and each set value is one of our literals.
              
              [{literal[5], literals[9]}, {literals[25]}]

            This example above represents the DNF formula: (literal5 OR literal9) AND literal25
            
            Uses max clauses, randomly picks 
        """
        # Create a list of ants with random DNF formulas

        for ant in self.ants:
            buy_dnf = self.generate_random_dnf_expression(True)
            sell_dnf = self.generate_random_dnf_expression(False)
            ant.buy_dnf = buy_dnf
            ant.sell_dnf = sell_dnf
        

    def generate_random_clause(self, is_buy):
        clause_length = max(1, int(random.gauss(self.average_literals, self.average_literals / 2)))
        clause = {self.generate_random_literal(is_buy)}
        while len(clause) < clause_length:
            clause.add(self.generate_random_literal(is_buy))
        return clause
    
    def generate_random_literal(self, is_buy):
        pheromones = self.buy_pheremones if is_buy else self.sell_pheremones
        return random.choices(self.literals, weights=pheromones, k=1)[0]

    def generate_random_dnf_expression(self, is_buy):
        num_clauses = max(1, int(random.gauss(self.average_clauses, self.average_clauses / 2)))
        expression = [self.generate_random_clause(is_buy)]
        for _ in range(1, num_clauses):
            expression.append(self.generate_random_clause(is_buy))
        return expression

    def construct_solution(self, ant):
        """
        This function updates the solution of the given ant by applying local search operators
        (e.g., adding/removing/changing literals or clauses) based on the pheromone probabilities
        The new solution should respect the max_clauses constraint.
        """
        # Modify the ant's DNF formula using probabilities derived from the pheromone matrix
        pass

    def evaluate_ant(self, ant):
        """
        Evaluate a DNF formula using the provided cost function.
        Result is subtracted by 100 to represent profit vs loss.
        """
        if ant.buy_dnf is None:
            return None
        return self.cost_function(ant.buy_dnf, ant.sell_dnf)-100
 

    def update_pheromone(self, profits): 
        """
            This function updates the pheromone levels of the buy and sell matrices based on the profit values of the ants' solutions.
            The evaporation_rate, alpha, and beta parameters should be used to control the pheromone update process.
            Normalise the matrices after updating the pheromone levels.
        """
        # Update the pheromone levels and apply evaporation
        pass

    def update_best_ant(self, profits):
        """
        This function updates our best ants array and adds the best ant if its better than all other best ants.
        It also updates our iteration if we haven't improved.
        """

        # Find the index of the ant with the best profit.
        best_ant_index = np.argmax(profits)

        # Check if the best ant has a better profit than the previous best ant
        # If best ants is empty, then we always add an ant.
        is_new_ant_better = (
            len(self.best_ants) == 0 or
            self.evaluate_ant(self.best_ants[-1]) < self.evaluate_ant(self.ants[best_ant_index])
        )

        # Update the best ants and current unimproved iteration
        if is_new_ant_better:
            ant_instance = copy.deepcopy(self.ants[best_ant_index])
            self.best_ants.append(ant_instance)
            self.current_unimproved_iteration = 0
        else:
            self.current_unimproved_iteration += 1


    def aco_algorithm(self):
        """
        Run the ACO algorithm for the given number of iterations and return the best DNF formula.
        """
        #test
        #return [{self.literals[1], self.literals[6]}, {self.literals[15]}], [{self.literals[12]}, {self.literals[6], self.literals[1]}]


        # Initialise ants
        self.initialise_ants()

        # Main loop
        

        for iteration in range(self.num_iterations):
            profits = []
            # Construct solutions
            for i, ant in enumerate(self.ants):
                print(f"Ant {i+1} of {len(self.ants)}")
                if iteration > 0:
                    self.construct_solution(ant)
                profits.append(self.evaluate_ant(ant))

            # Update pheromone levels
            self.update_pheromone(profits)

            # Update best ant if its the best we've ever found, 
            #   otherwise increment the iteration if no new best ant is found

            # print p to 2 decimal places
            # for p in profits:
            #     print("{:.2f}".format(p))

            self.update_best_ant(profits)


            # Check convergence 
            if self.current_unimproved_iteration == self.max_unimproved_iterations:
                break

        # Extract the best solution, which is the most recent best ant.
        best_buy_dnf_formula = self.best_ants[-1].buy_dnf
        best_sell_dnf_formula = self.best_ants[-1].sell_dnf

        return (best_buy_dnf_formula, best_sell_dnf_formula)


