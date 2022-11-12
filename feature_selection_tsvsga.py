import sys
import random
import time
import pandas as pd
import numpy as np
from queue import deque
import re
from pprint import pprint
import id3_algorithm
import time

"""
Configuration parameters
"""
FEATURE_SET_SIZE = 100      # Size of initial feature set
SUBSET_SIZE = 30            # Size of the selected feature set
TS_MOVES = 35               # Amount of putative solutions
LOOP_TRIES = 500            # How many potential moves to try in one move attempt
MAX_MOVES = 2000            # How many iterations the TS will have
TABU_LIST_SIZE = 250        # The amount of memorized tabooed solutions
ENABLE_IG_CACHE = True      # Cache the results of information gain calculation; disabling impacts performance
ACCEPT_THRESHOLD = 0.3      # Threshold value used in the acceptance criteria; set to 0 to disable
ENABLE_EXEC_LOG = False     # Enable detailed logs describing the algoritm flow
GENERATIONS = 2000          # Amount of generations in GA algorithm
POPULATION_SIZE = 10        # Number of feature subsets in one GA population
PARENTS_NUMBER = 6          # Number of top fitting feature subsets in one GA generation used as parents
MUTATION_RATE = 0.1         # Mutation rate in the Genetic Algorithm
MUTATION_DEGREE = 3         # The amount of features to be mutated



class DataLoader:
    """
    Provides functionality for loading datasets from file
    """
    def load(dataFile):
        """
        Loads data from a file in predefined format and returns it in a structured data set
        """
        #Import the dataset and define the feature as well as the target datasets / columns
        dataset = pd.read_table(dataFile, delim_whitespace=True, names=('A', 'Target'))
        dataset["A"] = dataset["A"].apply(list)
        #Initialize the list of feature names
        col_names = []
        for i in list(range(FEATURE_SET_SIZE)):
            col_names.append('feature_'+ str(i))
        #Apply feture names to the dataset
        dataset = pd.concat([dataset['A'].apply(pd.Series, index=col_names), dataset["Target"]], axis=1)
        return dataset
    
    def get_features(dataset):
        """
        Returns the complete list of features of a given data set
        """
        return dataset.columns[:-1]

    def sort_feature_set(features):
        """
        Sorts in a human expected way a feature set consisting from alpha-numeric values and returns it as a list
        """
        featList = list(features)
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(featList, key = alphanum_key)
    
    def feature_copy(data_frame, column_names):
        """
        Copies feature columns
        """
        subset = data_frame.loc[:, column_names]
        return subset



class IG:
    """
    Implements methods for calculating a dataset entropy and information gain of a selected feature set
    """
    def __init__(self):
        """
        Initialization method
        """
        if ENABLE_IG_CACHE:
            self.solutionsIgCache = dict()
            print("Calculation of IG initialized with cache...")
        else:
            print("IG initialized without cache...")
    
    def entropy(self, target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

    def info_gain(self, data, split_attribute_name, target_name="B"):
        """
        Calculate the information gain of a dataset. This function takes three parameters:
        1. data = The dataset for whose feature the IG should be calculated
        2. split_attribute_name = the name of the feature for which the information gain should be calculated
        3. target_name = the name of the target feature. The default for this example is "class"
        """
        #If cache enabled, first try find calculated information gain
        if ENABLE_IG_CACHE and split_attribute_name in self.solutionsIgCache.keys():
            return self.solutionsIgCache.get(split_attribute_name)

        #Calculate the entropy of the total dataset
        totalEntropy = self.entropy(data[target_name])
        
        ##Calculate the entropy of the dataset
        #Calculate the values and the corresponding counts for the split attribute 
        vals,counts = np.unique(data[split_attribute_name],return_counts=True)
        #Calculate the weighted entropy
        weightedEntropy = np.sum([(counts[i]/np.sum(counts))*self.entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
        #Calculate the information gain
        informationGain = totalEntropy - weightedEntropy
        # Save calculated information gain to cache if enabled
        if ENABLE_IG_CACHE:
            self.solutionsIgCache[split_attribute_name] = informationGain
        return informationGain

class TabuSearch:
    """
    Tabu Search implementation
    """
    def __init__(self, dataset):
        """
        Initialization method
        """
        self.tabuMemory = deque(maxlen = TABU_LIST_SIZE)
        self.ig = IG()
        self.aspirationMemory = set()
        self.data = dataset
        print("Tabu Search initialized...")
        
    def isTerminationCriteriaMet(self, solution):
        """
        Termination criteria is validating the score against a configured value, which can be set to 0 to disable
        """
        if ACCEPT_THRESHOLD > 0 and self.solutionFitness(solution) > ACCEPT_THRESHOLD:
            if ENABLE_EXEC_LOG:
                print("Termination criteria is met")
            return True
        return False

    def solutionFitness(self, solution):
        """
        Solution fitness score is calculated as a total of all information gains for each selected feature
        """
        return sum([self.ig.info_gain(self.data, feature, "Target") for feature in solution])

    def aspirationCriteria(self, solution):
        """
        A solution meets the aspiration criteria if it was never tried before
        """
        if ENABLE_EXEC_LOG and not solution in self.aspirationMemory:
            f = random.sample(list(solution), 3)
            # print("Aspiration criteria met with features {}, {}, {}...".format(f[0], f[1], f[2]))
        return not solution in self.aspirationMemory

    def tabuCriteria(self, solution):
        """
        Verifyes if a solution is not tabooed
        """
        if ENABLE_EXEC_LOG and solution in self.tabuMemory:
            f = random.sample(list(solution), 3)
            print("Tabooed solution with features {}, {}, {}...".format(f[0], f[1], f[2]))
        return not solution in self.tabuMemory

    def memorize(self, solution):
        """
        Memorizes current solution for further verification of tabu and aspiration criterias
        """
        self.tabuMemory.append(solution)
        self.aspirationMemory.add(frozenset(solution))

    def putativeNeighbors(self, solution, features):
        """
        Finds TS_MOVES putative solutions that satisfy aspiration and tabu criteria
        """
        putativeSolutions = [None] * TS_MOVES
        idx = 0
        count = 0
        while idx < TS_MOVES:
            count += 1
            if count > LOOP_TRIES:
                print("Unable to find more than {} neighbors".format(idx + 1))
            # Create a random altered solution off by only one feature from the current one
            alteredSolution = self.alterSolution(solution, features)
            # Save newly created solution if it satisfies aspiration criteria
            if self.aspirationCriteria(alteredSolution):
                putativeSolutions[idx] = alteredSolution
                idx += 1
            # Save newly created solution if it isn't tabooed or a duplicate of already saved new solution
            elif not putativeSolutions.__contains__(alteredSolution) and self.tabuCriteria(alteredSolution):
                putativeSolutions[idx] = alteredSolution
                idx += 1
        return putativeSolutions
    
    def alterSolution(self, solution, features):
        """
        Generates a new solution based on the provided by changing only one feature
        """
        alteredSolution = solution.copy()
        # Prepare the pool of features that can be used in new solution
        other_features = set(features) - solution
        # Remove a random feature
        alteredSolution.remove(random.choice(list(solution)))
        # Add a random feature from the pool
        alteredSolution.add(random.choice(list(other_features)))
        return alteredSolution


    def run(self):
        """
        Performs a run of Tabu Search based on initialized data set and returns the best found solution
        """
        print("Feature selection initiated.")
        if ENABLE_EXEC_LOG:
            print("\nexecution log:")
        features = DataLoader.get_features(self.data)
        self.bestSolution = set(random.sample(list(features), SUBSET_SIZE))
        self.memorize(self.bestSolution)
        self.currSolution = self.bestSolution
        
        step = 0
        while not self.isTerminationCriteriaMet(self.currSolution) and step < MAX_MOVES:
            step += 1
            # Get putative neighbors
            neighbors = self.putativeNeighbors(self.currSolution, features)
            bestFit = 0
            # Find the best solution from putative neighbors and makes it the current one
            for solution in neighbors:
                if self.solutionFitness(solution) > bestFit:
                    self.currSolution = solution
                    bestFit = self.solutionFitness(solution)
            # Memorize the current solution
            self.memorize(self.currSolution)
            # Verify if current solution is better than the best one, and saves the current as best, if true
            if self.solutionFitness(self.currSolution) > self.solutionFitness(self.bestSolution):
                self.bestSolution = self.currSolution

        # Return the best solution found
        return self.bestSolution

class GA:
    """
    Genetic Algorithm implemenetation for feature selection
    """
    def __init__(self, dataset):
        """
        Initialization method
        """
        self.ig = IG()
        self.data = dataset
        self.best_features = set()
        self.best_fintness = 0
        print("Genetic Algorithm initialized...")

    def init_population(self, features):
        population = list()
        count = 0
        for count in range(POPULATION_SIZE):
            population.append(set(random.sample(list(features), SUBSET_SIZE)))
        return population

    def select_children(self, population):
        """
        Single point crossover selection
        """
        next_generation = [set()] * POPULATION_SIZE
        for i in range(POPULATION_SIZE):
            parents = random.sample(population, 2)
            parent_1, parent_2 = parents[0], parents[1]
            common_features = set.intersection(parent_1, parent_2)
            diff_features = parent_1.union(parent_2).difference(common_features)
            delta = SUBSET_SIZE - len(common_features)
            next_generation[i] = common_features.union(set(random.sample(list(diff_features), delta)))
        return next_generation

    def displacement_mutation(self, population, features):
        """
        Mutation done through displacement

        """
        mutated = population.copy()
        for i in range(len(population)):
            if np.random.rand() < MUTATION_RATE:
                mutated[i] = mutated[i].difference(set(random.sample(mutated[i], MUTATION_DEGREE)))
                mutated[i] = mutated[i].union(set(random.sample(list(features), MUTATION_DEGREE)))
        return mutated
    
    def select_parents(self, population):
        """
        Random selector operator
        """
        fitness_stats = list()
        parents = [set()] * PARENTS_NUMBER
        for features in population:
            fitness = sum([self.ig.info_gain(self.data, feature, "Target") for feature in features])
            fitness_stats.append((fitness, features))
            if fitness > self.best_fintness:
                self.best_fintness = fitness
                self.best_features = features
                if ENABLE_EXEC_LOG:
                    print("Identified new best fintess = {}".format(fitness))
        fitness_stats.sort(key = lambda x: x[0])
        for i in range(PARENTS_NUMBER):
            parents[i] = fitness_stats[len(population) - i - 1][1]
        return parents

    def run(self):

        """
        Performs a run of Genetic Algorithm

        """
        population = self.init_population(DataLoader.get_features(self.data))
        features = DataLoader.get_features(self.data)
        for i in range(GENERATIONS):
            parents = self.select_parents(population)
            population = self.select_children(parents)
            population = self.displacement_mutation(population, features)
        print("Best solution with score {}\n{}".format(self.best_fintness, DataLoader.sort_feature_set(self.best_features)))

# Main execution 
start_ts = time.time()
print("*************************************************************")
print("*                TABU SEARCH IMPLEMENTATION                 *")
print("*************************************************************")
# Load data set
dataset = DataLoader.load("Training_Data.txt")

ts = TabuSearch(dataset)# Initialize Tabu Search instance


topSolution = DataLoader.sort_feature_set(ts.run())
print("\nBest solution with score {} :\n{}\n".format(ts.solutionFitness(topSolution), topSolution))

print("Tabu Search takes: {} seconds".format(time.time()-start_ts))

start_ga = time.time()

print("*******************************************************************")
print("*                GENETIC ALGORITHM IMPLEMENTATION                 *")
print("*******************************************************************")
    
ga = GA(dataset)# Initialize Genetic Algorithm instance
ga.run()

print("Tabu Search takes: {} seconds".format(time.time()-start_ga))

val_data = DataLoader.load("Validation_Data.txt")
test_data = DataLoader.load("Test_Data.txt")


'''

Copying features

'''

feature_set_ts = DataLoader.sort_feature_set(topSolution)

feature_subset_ts = DataLoader.feature_copy(dataset, feature_set_ts)
target = dataset["Target"]
feature_subset_ts = feature_subset_ts.join(target)


'''

Executing ID3 after getting feature subset from GA

'''
train_tree_ts = id3_algorithm.ID3(feature_subset_ts,feature_subset_ts,feature_subset_ts.columns[:-1])
pprint(train_tree_ts)

'''

Getting features from test data

'''


feature_subset_test_ts = DataLoader.feature_copy(test_data, feature_set_ts)
target = test_data["Target"]
feature_subset_test_ts = feature_subset_test_ts.join(target)


acc,rec,prec,tp,tn,fp,fn = id3_algorithm.test(feature_subset_test_ts,train_tree_ts)
print("*******************************************************")
print("*            PERFORMANCE WITH TABU SEARCH             *")
print("*******************************************************")
print(" accuracy = {}\n recall = {}\n precision = = {}\n TP = {}\n TN = = {}\n FP = {}\n FN = {}\n ".format(acc,rec,prec,tp,tn,fp,fn))

"""
Finding accuracies and comparing algorithms

"""




'''

Copying features

'''

feature_set_ga = DataLoader.sort_feature_set(ga.best_features)

feature_subset_ga = DataLoader.feature_copy(dataset, feature_set_ga)
target = dataset["Target"]
feature_subset_ga = feature_subset_ga.join(target)


'''

Executing ID3 after getting feature subset from GA

'''
train_tree_ga = id3_algorithm.ID3(feature_subset_ga,feature_subset_ga,feature_subset_ga.columns[:-1])
pprint(train_tree_ga)

'''

Getting features from test data

'''


feature_subset_test_ga = DataLoader.feature_copy(test_data, feature_set_ga)
target = test_data["Target"]
feature_subset_test_ga = feature_subset_test_ga.join(target)


acc_ga,rec_ga,prec_ga,tp_ga,tn_ga,fp_ga,fn_ga = id3_algorithm.test(feature_subset_test_ga,train_tree_ga)
print("*******************************************************")
print("*          PERFORMANCE WITH GENETIC ALGORITHM         *")
print("*******************************************************")
print(" accuracy = {}\n recall = {}\n precision = = {}\n TP = {}\n TN = = {}\n FP = {}\n FN = {}\n ".format(acc_ga,rec_ga,prec_ga,tp_ga,tn_ga,fp_ga,fn_ga))

"""
Converting to ones and zeros for chi square test. A = 1 and True = 1, 0 otherwise

"""
conv_TS = id3_algorithm.convertData(test_data)
feature_subset_TS = DataLoader.feature_copy(conv_TS, topSolution)
target = test_data["Target"]
feature_subset_TS = feature_subset_TS.join(target)
feature_subset_TS

"""
Similarly for GA

"""

conv_GA = id3_algorithm.convertData(test_data)
feature_subset_GA = DataLoader.feature_copy(conv_TS, DataLoader.sort_feature_set(ga.best_features))
feature_subset_GA = feature_subset_GA.join(target)
feature_subset_GA

"""

Now to get chi square test we use sklearns method
"""

from sklearn.feature_selection import chi2

X_TS = feature_subset_TS.drop('Target',axis=1)
y_TS = feature_subset_TS['Target']
chi_scores_TS = chi2(X_TS,y_TS)
chiSquareVal_TS = chi_scores_TS[0]
pVal_TS = chi_scores_TS[1]


chi_scores_TS

"""
Similarly for GA

"""

X_GA = feature_subset_GA.drop('Target',axis=1)
y_GA = feature_subset_GA['Target']
chi_scores_GA = chi2(X_TS,y_TS)
chiSquareVal_GA = chi_scores_GA[0]
pVal_GA = chi_scores_GA[1]

"""
Plotting for TS

"""
chi_TS = pd.Series(chiSquareVal_TS,index = X_TS.columns)
chi_TS.sort_values(ascending = False , inplace = True)
chi_TS.plot.density()

"""
Plotting for GA

"""

chi_GA = pd.Series(chiSquareVal_GA,index = X_GA.columns)
chi_GA.sort_values(ascending = False , inplace = True)
chi_GA.plot.density()

