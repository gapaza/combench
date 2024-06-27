import multiprocessing
import time
import math
from combench.models import truss

# Get number of available CPUs
num_cpus = multiprocessing.cpu_count()

class EvaluationProcessManager:

    def __init__(self, num_processes=num_cpus):
        self.num_processes = num_processes
        self.procs = []
        self.init_processes()

    def init_processes(self):
        self.procs = []
        for x in range(self.num_processes):
            request_queue = multiprocessing.Queue()
            response_queue = multiprocessing.Queue()
            eval_proc = EvaluationProcess2(request_queue, response_queue)
            eval_proc.start()
            self.procs.append([
                eval_proc, request_queue, response_queue
            ])

    def evaluate(self, problems, designs):
        if len(problems) != len(designs):
            raise ValueError('Number of problems and designs must be the same')

        # Zip problems and designs
        requests = list(zip(problems, designs))

        # Decompose designs into chunks
        chunk_size = math.ceil(len(designs) / self.num_processes)
        chunks = [requests[i:i + chunk_size] for i in range(0, len(requests), chunk_size)]

        r_queues = []
        for i, chunk in enumerate(chunks):
            self.procs[i][1].put(chunk)
            r_queues.append(self.procs[i][2])

        evals = []
        for r_queue in r_queues:
            evals += r_queue.get()
        return evals



class EvaluationProcess2(multiprocessing.Process):
    def __init__(self, request_queue, response_queue):
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.stop_event = multiprocessing.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Wait for a new request

                eval_request = self.request_queue.get()

                if eval_request is None:  # Use None to signal the process to stop
                    self.stop_event.set()
                    break

                # Perform the evaluation (replace this with actual evaluation logic)
                result = self.evaluate(eval_request)

                # Put the result in the response queue
                self.response_queue.put(result)
            except multiprocessing.queues.Empty:
                continue

    def evaluate(self, chunk):
        results = []
        for ch_dp in chunk:
            problem, design = ch_dp
            results.append(self.eval(problem, design))
        return results


    def eval(self, problem, design, normalize=True):
        stiff_vals = truss.eval_stiffness(problem, design, normalize=True)
        stiff = stiff_vals[0] * -1.0  # maximize stiffness_old
        if stiff == 0:
            volfrac = 1.0
        else:
            volfrac = truss.eval_volfrac(problem, design, normalize=True)
        return stiff, volfrac

    def stop(self):
        self.stop_event.set()


class EvaluationProcess(multiprocessing.Process):
    def __init__(self, request_queue, response_queue):
        super().__init__()
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.stop_event = multiprocessing.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                # Wait for a new request

                eval_request = self.request_queue.get()

                if eval_request is None:  # Use None to signal the process to stop
                    self.stop_event.set()
                    break

                # Perform the evaluation (replace this with actual evaluation logic)
                result = self.evaluate(eval_request)

                # Put the result in the response queue
                self.response_queue.put(result)
            except multiprocessing.queues.Empty:
                continue

    def evaluate(self, eval_request):
        problem, chunk = eval_request
        # print('PROC EVALUATING', problem, chunk)
        # results = [[0, 1] for x in range(len(chunk))]
        results = []
        for design in chunk:
            results.append(self.eval(problem, design))
        return results


    def eval(self, problem, design, normalize=True):
        stiff_vals = truss.eval_stiffness(problem, design, normalize=True)
        stiff = stiff_vals[0] * -1.0  # maximize stiffness_old
        if stiff == 0:
            volfrac = 1.0
        else:
            volfrac = truss.eval_volfrac(problem, design, normalize=True)
        return stiff, volfrac

    def stop(self):
        self.stop_event.set()










