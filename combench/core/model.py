from abc import ABC, abstractmethod
import random
import os
import config
import json

# imports for salt string
import string


class Model(ABC):

    def __init__(self, problem_formulation):
        self.problem_formulation = problem_formulation  # Dictionary containing the defined problem formulation
        if 'name' not in self.problem_formulation:
            # create salt string
            salt = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            print('Problem name not provided. Using salt string: {}'.format(salt))
            self.problem_formulation['name'] = salt
        self.model_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        self.problem_store_path = os.path.join(config.database_dir, self.problem_formulation['name'])
        self.problem_store = self.load_problem_store()
        self.norms = None
        self.pool = None

    @abstractmethod
    def load_norms(self):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    # The encoding scheme is problem specific, so this method is left abstract
    @abstractmethod
    def random_design(self):
        pass


    def load_problem_store(self):
        if os.path.exists(self.problem_store_path):
            with open(self.problem_store_path, 'r') as f:
                problem_data = json.load(f)
        else:
            problem_data = {}
        return problem_data

    def save_problem_store(self):
        with open(self.problem_store_path, 'w') as f:
            json.dump(self.problem_store, f)



