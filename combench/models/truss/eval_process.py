import multiprocessing
import time
import math
from combench.models import truss
import json

# Get number of available CPUs
num_cpus = multiprocessing.cpu_count()

class EvaluationProcessManager:

    def __init__(self, num_processes=num_cpus):
        self.num_processes = num_processes
        self.procs = []
        self.init_processes()
        self.eval_store = {}

    def shutdown(self):
        for proc in self.procs:
            # print('Shutting down process')
            proc[1].put(None)
            proc[0].join()


    def init_processes(self):
        self.procs = []
        for x in range(self.num_processes):
            request_queue = multiprocessing.Queue()
            response_queue = multiprocessing.Queue()
            eval_proc = EvaluationProcess3(request_queue, response_queue)
            eval_proc.start()
            self.procs.append([
                eval_proc, request_queue, response_queue
            ])

    def evaluate_store(self, problems, designs):
        total_results = len(designs)
        final_results = [None for x in range(total_results)]
        designs_to_eval = []
        designs_to_eval_idx = []
        designs_alr_eval = []
        designs_alr_eval_idx = []
        for idx, (problem, design) in enumerate(zip(problems, designs)):
            problem_str = json.dumps(problem)
            design_str = ''.join([str(x) for x in design])
            if problem_str not in self.eval_store:
                self.eval_store[problem_str] = {}
                designs_to_eval.append((problem, design))
                designs_to_eval_idx.append(idx)
            elif design_str not in self.eval_store[problem_str]:
                designs_to_eval.append((problem, design))
                designs_to_eval_idx.append(idx)
            else:
                objs = self.eval_store[problem_str][design_str]
                designs_alr_eval.append(objs)
                designs_alr_eval_idx.append(idx)
                final_results[idx] = objs

        if len(designs_to_eval) > 0:
            probs = [x[0] for x in designs_to_eval]
            des = [x[1] for x in designs_to_eval]
            evals = self.evaluate(probs, des)
            for idx, (problem, design) in enumerate(zip(probs, des)):
                problem_str = json.dumps(problem)
                design_str = ''.join([str(x) for x in design])
                objs = evals[idx]
                self.eval_store[problem_str][design_str] = evals[idx]
                final_idx = designs_to_eval_idx[idx]
                final_results[final_idx] = objs

        return final_results

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






class EvaluationProcess3(multiprocessing.Process):
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
        # design = truss.rep.get_bit_list_from_node_seq(problem, design)
        if isinstance(design, list) and 1 not in design:
            stiff = 0
        else:
            stiff_vals = truss.eval_stiffness(problem, design, normalize=True)
            stiff = stiff_vals[0] * -1.0  # maximize stiffness_old
        if stiff == 0:
            volfrac = 1.0
        else:
            volfrac = truss.eval_volfrac(problem, design, normalize=True)
        return stiff, volfrac

    def stop(self):
        self.stop_event.set()



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
        if isinstance(design, list) and 1 not in design:
            stiff = 0
        else:
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










