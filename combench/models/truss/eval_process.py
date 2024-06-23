import multiprocessing
import time

from combench.models import truss

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
        volfrac = truss.eval_volfrac(problem, design)
        return stiff, volfrac

    def stop(self):
        self.stop_event.set()










