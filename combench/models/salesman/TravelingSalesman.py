import math
from combench.core.model import Model
import random
from copy import deepcopy





class TravelingSalesman(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.cities = problem_formulation.get('cities', [])
        self.cities = self.normalize_coords(self.cities)
        self.costs = problem_formulation.get('costs', [])
        self.costs = self.normalize_coords(self.costs)

        self.num_cities = len(self.cities)
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))

    def normalize_coords(self, coords):
        cities_x = [city[0] for city in coords]
        cities_y = [city[1] for city in coords]

        # Find min and max for x and y
        min_x, max_x = min(cities_x), max(cities_x)
        min_y, max_y = min(cities_y), max(cities_y)

        # Normalize x and y values
        range_x = max_x - min_x
        range_y = max_y - min_y

        normalized_coords = [
            ((x - min_x) / range_x, (y - min_y) / range_y) for x, y in coords
        ]
        return normalized_coords


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

    def is_valid(self, city_list):
        unique_cities_visited = set()
        for i in range(len(city_list)):
            city = city_list[i]
            if city in unique_cities_visited:
                return False
            unique_cities_visited.add(city)
        return True

    def evaluate(self, design, normalize=True):
        # if self.is_valid(design) is False:
        #     if normalize is True:
        #         return 1.0, 1.0, False
        #     return 1e10, 1e10, False
        total_distance, total_cost = self._evaluate(design)
        if normalize is True:
            if total_distance == 1e10:
                total_cost = 1.0
                total_distance = 1.0
            else:
                distance_norm, cost_norm = self.norms
                total_distance /= distance_norm
                total_cost /= cost_norm
        return total_distance, total_cost, True

    def _evaluate(self, design_copy):
        """
        Evaluate a given tour in terms of multiple objectives:
        - Minimize total distance traveled
        - Minimize the number of cities visited (if applicable)

        :param design: List of integers representing the order of cities to visit
        :return: A tuple containing the total distance and number of cities visited
        """
        design = deepcopy(design_copy)

        # Evaluate tour distance
        total_distance = 0
        num_cities_visited = len(design)
        if num_cities_visited > self.num_cities:
            raise ValueError('The number of cities visited cannot exceed the total number of cities')

        # If any cities are unvisited, determine which
        unvisited_cities = set(range(self.num_cities)) - set(design)
        if unvisited_cities:
            design.extend(list(unvisited_cities))
        # print('Design: {}'.format(design))

        # Calculate the total distance traveled
        unique_cities_visited = set()
        # unique_edges_traversed = set()
        for i in range(len(design) - 1):
            city1 = self.cities[design[i]]
            city2 = self.cities[design[i + 1]]
            dist = math.sqrt(abs(city1[0] - city2[0]) ** 2) + (abs(city1[1] - city2[1]) ** 2)
            if city2 in unique_cities_visited:
                dist *= 2
            if city1 == city2:
                dist += 0.5
            # edge = (design[i], design[i + 1])
            # rev_edge = (design[i + 1], design[i])
            # if edge in unique_edges_traversed or rev_edge in unique_edges_traversed:
            #     dist *= 2
            # else:
            #     unique_edges_traversed.add(edge)
            #     unique_edges_traversed.add(rev_edge)
            unique_cities_visited.add(city1)
            total_distance += dist
        # print('Tour distance: {}'.format(total_distance))

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
from combench.models.salesman import problem2

if __name__ == '__main__':
    model = TravelingSalesman(problem2)

    # Evaluate a tour
    tour = [0, 4, 2, 3, 2]
    tour2 = [
            2,
            4,
            0,
            1,
            2
        ]
    total_distance, total_cost, is_feasible = model.evaluate(tour)
    print(f'Total distance: {total_distance}')
    print(f'Total cost: {total_cost}\n\n')
    total_distance, total_cost, is_feasible = model.evaluate(tour2)
    print(f'Total distance: {total_distance}')
    print(f'Total cost: {total_cost}')