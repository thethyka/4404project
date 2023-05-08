import numpy as np
import copy
import random 




# Define the Ant class to represent an individual ant in the ACO algorithm
class Ant:
    def __init__(self, buy_dnf=None, sell_dnf=None):
        # Initialize the ant with optional buy and sell DNFs
        self.buy_dnf = buy_dnf
        self.sell_dnf = sell_dnf
        self.money = 0
    def __str__(self):
        return "Buy DNF: " + str(self.buy_dnf) + "\nSell DNF: " + str(self.sell_dnf)

# Define the ACOOptimiser class to run the ACO algorithm
class ACOOptimiser:
    def __init__(self, cost_function, keys, acoparams):
        # Initialize the ACOOptimiser with a cost function and literal keys

        # Set the parameters for the ACO algorithm
        self.cost_function = cost_function
        self.literals = list(keys)



        # Controls the influence of how much money the ant made on the pheremone
        self.alpha = acoparams[0]
        # Controls the influence of rarer literals being more important
        self.beta = acoparams[1]
        # Determines the rate at which pheromone trails evaporate over time
        self.evaporation_rate = acoparams[2]

        # Represents the average number of clauses in the DNF formulas
        self.average_clauses = acoparams[3]
        # Represents the average number of literals in each clause of the DNF formulas
        self.average_literals = acoparams[4]

        self.clause_mutation_rate = 0.5

        # Represents the number of ants in the ant colony optimization algorithm
        self.num_ants = acoparams[5]
        # Represents the number of iterations the ant colony optimization algorithm will run for
        self.num_iterations = acoparams[6]


        # All our ants for the ACO algorithm
        self.ants = [Ant() for _ in range(self.num_ants)]

        # A list of our best-found ants while optimising.
        self.best_ants = []

        # Set the parameters for checking convergence
        # Current unimproved iteration set to 0 if a best ant is added to list.
        self.max_unimproved_iterations = 25
        self.current_unimproved_iteration = 0


        # A list of literals for each key, along with their negations
        new_literals = []
        for literal in self.literals:
            new_literals.append((literal, True))
            new_literals.append((literal, False))
        self.literals = new_literals


        # Initialize the pheromone values for buy and sell literals
        self.buy_pheromones = [1 for _ in range(len(self.literals))]
        self.sell_pheromones = [1 for _ in range(len(self.literals))]
        self.normalise_pheromones()

        self.historical_buy_pheromones = []
        self.historical_sell_pheromones = []





    def normalise_pheromones(self):
        self.buy_pheromones = [x/sum(self.buy_pheromones) for x in self.buy_pheromones]
        self.sell_pheromones = [x/sum(self.sell_pheromones) for x in self.sell_pheromones]

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
        pheromones = self.buy_pheromones if is_buy else self.sell_pheromones
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
        
        #can make probabilites only be calculated each iteration if too slow
        def add_literal(clause, is_buy):
            if len(clause) < len(self.literals):
                new_literal = self.generate_random_literal(is_buy)
                while new_literal in clause:
                    new_literal = self.generate_random_literal(is_buy)
                clause.add(new_literal)

        def remove_literal(clause):
            if len(clause) > 1:
                clause.remove(random.choice(list(clause)))

        def change_literal(clause, is_buy):
            if len(clause) > 0:
                literal_to_replace = random.choice(list(clause))
                clause.remove(literal_to_replace)
                add_literal(clause, is_buy)

    
        for dnf_type in ["buy", "sell"]:
            dnf = ant.buy_dnf if dnf_type == "buy" else ant.sell_dnf
            for clause in dnf:

                if random.random() > self.clause_mutation_rate:
                    continue

                operator = np.random.choice(["add", "remove", "change"])
                if operator == "add":
                    add_literal(clause, dnf_type == "buy")
                elif operator == "remove":
                    remove_literal(clause)
                elif operator == "change":
                    change_literal(clause, dnf_type == "buy")
        
    

    def evaluate_ant(self, ant):
        """
        Evaluate a DNF formula using the provided cost function.
        Result is subtracted by 100 to represent profit vs loss.
        """
        if ant.buy_dnf is None:
            return None
        
        profit = self.cost_function(ant.buy_dnf, ant.sell_dnf) - 100

        ant.money = profit + 100

        return profit
 

    def update_pheromones(self): 
        """
            This function updates the pheromone levels of the buy and sell matrices based on the profit values of the ants' solutions.
            The evaporation_rate, alpha, and beta parameters should be used to control the pheromone update process.
            Normalise the matrices after updating the pheromone levels.
        """
        # Update the pheromone levels and apply evaporation
        # Calculate the total money made by all ants.
        self.historical_buy_pheromones.append(copy.deepcopy(self.buy_pheromones))
        self.historical_sell_pheromones.append(copy.deepcopy(self.sell_pheromones))

        total_money = sum(ant.money for ant in self.ants)

        # Update pheromone levels for both buy and sell lists
        for dnf_type in ["buy", "sell"]:
            pheromones = self.buy_pheromones if dnf_type == "buy" else self.sell_pheromones
            for i in range(len(pheromones)):
                # Evaporate the pheromone
                pheromones[i] *= (1 - self.evaporation_rate)

                # Update the pheromone based on the ants' profits
                for ant in self.ants:
                    dnf = ant.buy_dnf if dnf_type == "buy" else ant.sell_dnf
                    literal_presence = sum([1 for clause in dnf if self.literals[i] in clause])
                    if literal_presence > 0:
                        contribution = (ant.money / total_money) ** self.alpha * (1 / (1 + literal_presence)) ** self.beta
                        pheromones[i] += contribution

        # Normalize the pheromone levels in the list
        self.normalise_pheromones()

        

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
                # print(f"Ant {i+1} of {len(self.ants)} for iteration {iteration+1} of {self.num_iterations}")
                if iteration > 0:
                    self.construct_solution(ant)
                profits.append(self.evaluate_ant(ant))

            


            # Update pheromone levels
            
            self.update_pheromones()

            # Update best ant if its the best we've ever found, 
            #   otherwise increment the iteration if no new best ant is found

            self.update_best_ant(profits)


            # Check convergence 
            if self.current_unimproved_iteration == self.max_unimproved_iterations:
                break

        # Extract the best solutions 

        # store final best pheremone matrices
        self.historical_buy_pheromones.append(copy.deepcopy(self.buy_pheromones))
        self.historical_sell_pheromones.append(copy.deepcopy(self.sell_pheromones))

        return (self.best_ants, self.historical_buy_pheromones, self.historical_sell_pheromones)


