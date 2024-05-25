import config
from copy import deepcopy
import math
import random
from combench.models.utils import random_binary_design
from tqdm import tqdm
from combench.core.model import Model
import multiprocessing

# spawn new processes
# from multiprocessing import set_start_method
# set_start_method('fork', force=True)

class GncModel(Model):

    def __init__(self, problem_formulation):
        super().__init__(problem_formulation)
        self.sensors = problem_formulation['sensors']['reliabilities']
        self.computers = problem_formulation['computers']['reliabilities']
        self.actuators = problem_formulation['actuators']['reliabilities']
        self.connection_reliability = problem_formulation['connection_reliability']

        self.sensors_weights = problem_formulation['sensors']['weights']
        self.computers_weights = problem_formulation['computers']['weights']
        self.actuators_weights = problem_formulation['actuators']['weights']
        self.connection_weight = problem_formulation['connection_weight']
        self.norms = self.load_norms()
        print('Norm values: {}'.format(self.norms))


    def load_norms(self):
        if 'norms' in self.problem_store:
            return self.problem_store['norms']

        # Calculate the norms
        random_designs = [self.random_design() for _ in range(1000)]
        # print(random_designs[0])
        objs_batch = self.evaluate_batch(random_designs, normalize=False)
        evals = objs_batch
        max_reliability = min([evals[i][0] for i in range(len(evals))])
        max_mass = max([evals[i][1] for i in range(len(evals))])
        reliability_norm = abs(max_reliability) * 1.1
        mass_norm = max_mass * 1.1
        self.problem_store['norms'] = [reliability_norm, mass_norm]
        self.save_problem_store()
        return [reliability_norm, mass_norm]


    def random_design(self):
        sca_len = len(self.sensors) * len(self.computers)
        caa_len = len(self.computers) * len(self.actuators)
        n = sca_len + caa_len
        design = random_binary_design(n)
        return design


    # ----------------------------------------------
    # Evaluate
    # ----------------------------------------------

    def evaluate_batch(self, designs, normalize=False):
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(pool.imap(self._evaluate_single_design, [(design, normalize) for design in designs]))
        return results

    def _evaluate_single_design(self, args):
        design, normalize = args
        return self.evaluate(design, normalize)

    def evaluate(self, design, normalize=False):
        sca_len = len(self.sensors) * len(self.computers)
        caa_len = len(self.computers) * len(self.actuators)
        sensor_computer_assignment = design[:sca_len]
        computer_actuator_assignment = design[sca_len:]
        sensor_computer_assignment = ''.join([str(bit) for bit in sensor_computer_assignment])
        computer_actuator_assignment = ''.join([str(bit) for bit in computer_actuator_assignment])
        objectives = self._evaluate(sensor_computer_assignment, computer_actuator_assignment, scale=True)
        reliability, mass = objectives
        if normalize is True:
            reliability_norm, mass_norm = self.norms
            reliability = reliability / reliability_norm
            mass = mass / mass_norm
        # print("Reliability: ", reliability, " Mass: ", mass)
        return -reliability, mass

    def _evaluate(self, sensor_computer_assignment, computer_actuator_assignment, scale=False):
        # print("Evaluating design: ", sensor_computer_assignment, computer_actuator_assignment)
        sca_len = len(self.sensors) * len(self.computers)
        caa_len = len(self.computers) * len(self.actuators)
        if len(sensor_computer_assignment) != sca_len or len(computer_actuator_assignment) != caa_len:
            raise ValueError("Assignment strings must be of length: ", sca_len, caa_len, " but are: ", len(sensor_computer_assignment), len(computer_actuator_assignment))

        self.sensor_computer_assignment = [int(bit) for bit in sensor_computer_assignment]
        self.computer_actuator_assignment = [int(bit) for bit in computer_actuator_assignment]
        self.design_sensors = set()
        self.design_computers = set()
        self.design_actuators = set()
        counter = 0
        for c_idx in range(len(self.computers)):
            for s_idx in range(len(self.sensors)):
                bit = int(sensor_computer_assignment[counter])
                if bit == 1:
                    self.design_sensors.add(s_idx)
                    self.design_computers.add(c_idx)
                counter += 1
        counter = 0
        for a_idx in range(len(self.actuators)):
            for c_idx in range(len(self.computers)):
                bit = int(computer_actuator_assignment[counter])
                if bit == 1:
                    self.design_actuators.add(a_idx)
                counter += 1
        self.design_sensors_probs = [self.sensors[x] for x in self.design_sensors]
        self.design_computers_probs = [self.computers[x] for x in self.design_computers]
        self.design_actuators_probs = [self.actuators[x] for x in self.design_actuators]
        # print("Design sensors: ", self.design_sensors, " Design computers: ", self.design_computers, " Design actuators: ", self.design_actuators)
        self.sensor_failures = self.generate_component_failures(self.design_sensors)
        self.computer_failures = self.generate_component_failures(self.design_computers)
        self.actuator_failures = self.generate_component_failures(self.design_actuators)

        self.sensor_computer_connection_failures, self.sc_conn_list = self.generate_connection_failure(
            sensor_computer_assignment)
        self.computer_actuator_connection_failures, self.ca_conn_list = self.generate_connection_failure(
            computer_actuator_assignment)


        # Compute reliability
        reliability = self.evaluate_reliability(scale=scale)

        # Compute mass
        mass = self.evaluate_mass()

        # Always return objectives as they are
        # - flipping values for minimization / maximization is done in the design object
        objectives = [reliability, mass]
        return objectives

    # ----------------------------------------------
    # Helper functions
    # ----------------------------------------------

    def evaluate_mass(self):
        design_sensor_weights = [self.sensors_weights[x] for x in self.design_sensors]
        design_computer_weights = [self.computers_weights[x] for x in self.design_computers]
        design_actuator_weights = [self.actuators_weights[x] for x in self.design_actuators]
        sensor_computer_connection_weights = self.connection_weight * sum(self.sensor_computer_assignment)
        computer_actuator_connection_weights = self.connection_weight * sum(self.computer_actuator_assignment)

        total_mass = sum(design_sensor_weights) + sum(design_computer_weights) + sum(design_actuator_weights) + sensor_computer_connection_weights + computer_actuator_connection_weights
        return total_mass

    def evaluate_reliability(self, scale=False):
        reliability = 0

        # Iterate through all possible failure scenarios
        # - This is a combinatorial problem, so we need to iterate through all possible combinations of failures
        valid_configs = 0
        for sensor_failure in self.sensor_failures:
            sensor_prob = self.component_failure_prob(self.design_sensors_probs, sensor_failure)
            # print(sensor_failure, sensor_prob)

            # rebuild full sensor bitlst
            sensor_bitlst = []
            count = 0
            for i in range(len(self.sensors)):
                if i in self.design_sensors:
                    if sensor_failure[count] == 1:
                        sensor_bitlst.append(1)
                    else:
                        sensor_bitlst.append(0)
                    count += 1
                else:
                    sensor_bitlst.append(0)

            for scc_idx, sensor_computer_connection_failure in enumerate(self.sensor_computer_connection_failures):
                sensor_computer_connection_prob = self.connection_failure_prob(self.sc_conn_list[scc_idx])
                # print(self.sc_conn_list[scc_idx], sensor_computer_connection_prob)

                for computer_failure in self.computer_failures:
                    computer_prob = self.component_failure_prob(self.design_computers_probs, computer_failure)

                    # rebuild full computer bitlst
                    computer_bitlst = []
                    count = 0
                    for i in range(len(self.computers)):
                        if i in self.design_computers:
                            if computer_failure[count] == 1:
                                computer_bitlst.append(1)
                            else:
                                computer_bitlst.append(0)
                            count += 1
                        else:
                            computer_bitlst.append(0)

                    for cac_idx, computer_actuator_connection_failure in enumerate(self.computer_actuator_connection_failures):
                        computer_actuator_connection_prob = self.connection_failure_prob(self.ca_conn_list[cac_idx])

                        for actuator_failure in self.actuator_failures:
                            actuator_prob = self.component_failure_prob(self.design_actuators_probs, actuator_failure)

                            # rebuild full actuator bitlst
                            actuator_bitlst = []
                            count = 0
                            for i in range(len(self.actuators)):
                                if i in self.design_actuators:
                                    if actuator_failure[count] == 1:
                                        actuator_bitlst.append(1)
                                    else:
                                        actuator_bitlst.append(0)
                                    count += 1
                                else:
                                    actuator_bitlst.append(0)

                            system_status = self.eval_system(sensor_bitlst, sensor_computer_connection_failure, computer_bitlst, computer_actuator_connection_failure, actuator_bitlst)

                            if system_status is True:
                                valid_configs += 1
                                reliability += (sensor_prob * sensor_computer_connection_prob * computer_prob * computer_actuator_connection_prob * actuator_prob)


        # print("Reliability: ", reliability)
        # print("Valid configs: ", valid_configs)
        # Compute result: double result = -(Math.log10(1.0-reliability));
        if scale:
            result = -math.log10(1.0 - reliability)
        else:
            result = reliability
        return result

    def generate_active_component_failures(self, components):
        # Components is a binary list denoting which components are active
        idx_active = [i for i, bit in enumerate(components) if bit == 1]
        active_failures = self.generate_component_failures(idx_active)

        all_full_failures = []
        for active_failure in active_failures:
            full_failure = [0 for i in range(len(components))]
            for i, idx in enumerate(idx_active):
                full_failure[idx] = active_failure[i]
            all_full_failures.append(full_failure)
        return all_full_failures

    def generate_component_failures(self, component):
        """
        Generate all possible binary strings of length n, where n is the number of components in the system.

        If a component isn't used in the design, treat it as having failed in the representation.
        This way, a static representation can be used when evaluating reliability


        :param component: An array list of doubles
        :return: List of binary strings representing component failures.
        """
        n = len(component)
        binary_strings = enumerate_binary_strings(n)
        binary_lists = [[int(bit) for bit in binary_string] for binary_string in binary_strings]
        return binary_lists

    def generate_connection_failure(self, bitstr):
        bitlist = [int(bit) for bit in bitstr]

        active_indices = []
        for i, bit in enumerate(bitlist):
            if bit == 1:
                active_indices.append(i)

        failure_scenarios = enumerate_binary_strings(len(active_indices))
        failure_lists = [[int(bit) for bit in binary_string] for binary_string in failure_scenarios]
        f_lst_copy = deepcopy(failure_lists)

        connection_failure_chromosomes = []
        for failure_list in failure_lists:
            connection_failure_chromosome = deepcopy(bitlist)
            for i, bit in enumerate(bitlist):
                if i in active_indices:
                    connection_failure_chromosome[i] = failure_list.pop(0)
            connection_failure_chromosomes.append(connection_failure_chromosome)

        return connection_failure_chromosomes, f_lst_copy

    def eval_system(self, sensor_failure, sensor_computer_connection_failure, computer_failure, computer_actuator_connection_failure, actuator_failure):

        # Iterate over computer and sensors
        idx = 0
        val_comps = []
        for c_idx, c_bit in enumerate(computer_failure):
            c_bit = computer_failure[c_idx]
            val_comp_bits = []
            for s_idx, s_bit in enumerate(sensor_failure):
                s_bit = sensor_failure[s_idx]
                conn_bit = sensor_computer_connection_failure[idx]
                val_bit = c_bit * s_bit * conn_bit
                val_comp_bits.append(val_bit)
            if sum(val_comp_bits) > 0:
                val_comps.append(1)
            else:
                val_comps.append(0)
        # print("Val comps: ", val_comps)

        # Iterate over actuators and computers
        idx = 0
        val_actuators = []
        for a_idx, a_bit in enumerate(actuator_failure):
            a_bit = actuator_failure[a_idx]
            val_actuator_bits = []
            for c_idx, c_bit in enumerate(computer_failure):
                c_bit = computer_failure[c_idx]
                conn_bit = computer_actuator_connection_failure[idx]
                val_bit = a_bit * c_bit * conn_bit
                val_actuator_bits.append(val_bit)
            if sum(val_actuator_bits) > 0:
                val_actuators.append(1)
            else:
                val_actuators.append(0)
        # print("Val actuators: ", val_actuators)

        if sum(val_actuators) > 0:
            return True
        else:
            return False

    def component_failure_prob(self, components, bitlst):
        prob = 1
        for i, component_prob in enumerate(components):
            bit = bitlst[i]
            if bit == 1:
                prob *= component_prob
            else:
                prob *= (1 - component_prob)
        return prob

    def connection_failure_prob(self, bitlst):
        prob = 1
        for bit in bitlst:
            if bit == 1:
                prob *= self.connection_reliability
            else:
                prob *= (1 - self.connection_reliability)
        return prob


def enumerate_binary_strings(n):
    """
        Generate all binary strings of length n.

        :param n: Length of the binary strings.
        :return: List of binary strings of length n.
        """
    if n < 1:
        return []

    result = []

    def backtrack(current):
        if len(current) == n:
            result.append(current)
            return
        backtrack(current + '0')
        backtrack(current + '1')

    backtrack('')
    return result



from combench.models.gnc import problem1


if __name__ == '__main__':


    gnc_model = GncModel(problem1)



    ### Evaluate objectives
    sensor_computer_assignment = '101001'
    computer_actuator_assignment = '110101'
    design = sensor_computer_assignment + computer_actuator_assignment
    design = [int(bit) for bit in design]
    objectives = gnc_model.evaluate(design)
    print("Reliability / Mass: ", objectives)







