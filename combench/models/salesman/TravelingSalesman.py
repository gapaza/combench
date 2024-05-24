import math
from combench.interfaces.model import Model
import random





class TravelingSalesman(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.cities = problem_formulation.get('cities', [])
        self.costs = problem_formulation.get('costs', [])
        self.num_cities = len(self.cities)
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))

    def load_norms(self):
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        # Calculate the norms
        random_designs = [self.random_design() for _ in range(10000)]
        evals = []
        for design in random_designs:
            objs = self.evaluate(design, normalize=False)
            if objs[0] == 1e10:
                continue
            evals.append(objs)
        max_distance = max([evals[i][0] for i in range(len(evals))])
        max_cost = max([evals[i][1] for i in range(len(evals))])
        distance_norm = max_distance * 1.1  # Add margin
        cost_norm = max_cost * 1.1  # Add margin
        self.problem_store['norms'] = [distance_norm, cost_norm]
        self.save_problem_store()
        return [distance_norm, cost_norm]

    def random_design(self):
        design = []
        for i in range(self.num_cities):
            city_num = random.randint(0, self.num_cities - 1)
            while city_num in design:
                city_num = random.randint(0, self.num_cities - 1)
            design.append(city_num)
        return design

    def evaluate(self, design, normalize=True):
        total_distance, total_cost = self._evaluate(design)
        if normalize is True:
            if total_distance == 1e10:
                total_cost = 1.0
                total_distance = 1.0
            else:
                distance_norm, cost_norm = self.norms
                total_distance /= distance_norm
                total_cost /= cost_norm
        return total_distance, total_cost

    def _evaluate(self, design):
        """
        Evaluate a given tour in terms of multiple objectives:
        - Minimize total distance traveled
        - Minimize the number of cities visited (if applicable)

        :param design: List of integers representing the order of cities to visit
        :return: A tuple containing the total distance and number of cities visited
        """

        # Evaluate tour distance
        total_distance = 0
        num_cities_visited = len(design)
        if num_cities_visited > self.num_cities:
            raise ValueError('The number of cities visited cannot exceed the total number of cities')

        # Calculate the total distance traveled
        unique_cities_visited = set()
        for i in range(num_cities_visited - 1):
            city1 = self.cities[design[i]]
            city2 = self.cities[design[i + 1]]
            if city1 in unique_cities_visited:  # return a very large number
                return 1e10, 1e10
            unique_cities_visited.add(city1)
            if city1 == city2:
                return 1e10, 1e10
            dist = math.sqrt(abs(city1[0] - city2[0])**2) + (abs(city1[1] - city2[1])**2)
            total_distance += dist

        # Add distance from the last city back to the first to complete the tour
        city1 = self.cities[design[-1]]
        city2 = self.cities[design[0]]
        dist = math.sqrt(abs(city1[0] - city2[0]) ** 2) + (abs(city1[1] - city2[1]) ** 2)
        total_distance += dist

        # Evaluate tour cost
        total_cost = 0
        for i in range(num_cities_visited - 1):
            cost1 = self.costs[design[i]]
            cost2 = self.costs[design[i + 1]]
            cost = math.sqrt(abs(cost1[0] - cost2[0]) ** 2) + (abs(cost1[1] - cost2[1]) ** 2)
            total_cost += cost

        # Add cost from the last city back to the first to complete the tour
        cost1 = self.costs[design[-1]]
        cost2 = self.costs[design[0]]
        cost = math.sqrt(abs(cost1[0] - cost2[0]) ** 2) + (abs(cost1[1] - cost2[1]) ** 2)
        total_cost += cost

        return total_distance, total_cost


from combench.models.salesman import problem1

if __name__ == '__main__':
    model = TravelingSalesman(problem1)

    # Evaluate a tour
    # tour = [0, 1, 2, 3, 4, 5, 6]
    # total_distance, total_cost = model.evaluate(tour)
    # print(f'Total distance: {total_distance}')
    # print(f'Total cost: {total_cost}')