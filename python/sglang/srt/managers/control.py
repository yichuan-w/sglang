import zmq
import zmq.asyncio
import queue
import random
import logging
import asyncio
from enum import Enum, auto

from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback
from sglang.srt.managers.router.worker import WorkerThread
from sglang.srt.managers.router.radix_cache_host import RadixCacheHost

logger = logging.getLogger("srt.controller")


class DispatchMethod(Enum):
    LOTTERY = auto()
    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()
    SHORTEST_LOCAL_QUEUE = auto()
    CACHE_AWARE_REQ_BALANCE = auto()
    CACHE_AWARE_MEM_BALANCE = auto()
    PRE_SCHEDULED = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.lower()
        if method == "lottery":
            return cls.LOTTERY
        elif method == "round_robin":
            return cls.ROUND_ROBIN
        elif method == "shortest_queue":
            return cls.SHORTEST_QUEUE
        elif method == "shortest_local_queue":
            return cls.SHORTEST_LOCAL_QUEUE
        elif method == "cache_aware_req_balance":
            return cls.CACHE_AWARE_REQ_BALANCE
        elif method == "cache_aware_mem_balance":
            return cls.CACHE_AWARE_MEM_BALANCE
        elif method == "pre_scheduled":
            return cls.PRE_SCHEDULED
        else:
            raise ValueError(f"Invalid dispatch method: {method}")


class Controller:

    def __init__(self, dispatch_method: str, server_args: ServerArgs,
                 port_args: PortArgs):
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        if self.dispatch_method == DispatchMethod.ROUND_ROBIN:
            self.round_robin_counter = 0
        elif self.dispatch_method in [
                DispatchMethod.CACHE_AWARE_REQ_BALANCE,
                DispatchMethod.CACHE_AWARE_MEM_BALANCE
        ]:
            self.radix_cache_host = RadixCacheHost()

        self.dispatch_lookup = {
            DispatchMethod.LOTTERY: self.lottery_worker,
            DispatchMethod.ROUND_ROBIN: self.round_robin_worker,
            DispatchMethod.SHORTEST_QUEUE: self.shortest_queue_worker,
            DispatchMethod.SHORTEST_LOCAL_QUEUE:
            self.shortest_local_queue_worker,
            DispatchMethod.CACHE_AWARE_REQ_BALANCE:
            self.cache_aware_req_balance_worker,
            DispatchMethod.CACHE_AWARE_MEM_BALANCE:
            self.cache_aware_mem_balance_worker,
            DispatchMethod.PRE_SCHEDULED: self.pre_scheduled_worker
        }
        self.dispatching = self.dispatch_lookup[self.dispatch_method]

        self.workers = {}
        for i in range(server_args.num_workers):
            self.workers[i] = {}
            self.workers[i]["request_queue"] = queue.Queue()
            self.workers[i]["client"] = WorkerThread(
                self.workers[i]["request_queue"], i, server_args, port_args)
            self.workers[i]["client"].start()

        # Init communication
        context = zmq.asyncio.Context()
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(
            f"tcp://127.0.0.1:{port_args.router_port}")

        # Init status
        self.recv_reqs = []
        # Init some configs

    def set_dispatch_method(self, dispatch_method: str):
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)
        self.dispatching = self.dispatch_lookup[self.dispatch_method]

    def register_worker(self, worker_addr: str):
        # todo: figure out what would be the best interface for a worker to register itself
        raise NotImplementedError

    def put_req_to_worker(self, worker_id, req, update_cache=False):
        self.workers[worker_id]["request_queue"].put(req)
        if update_cache:
            self.radix_cache_host.insert(tuple(req.input_ids), worker_id)

    def lottery_worker(self, input_requests):
        available_workers = list(self.workers.keys())
        for r in input_requests:
            self.put_req_to_worker(random.choice(available_workers), r)
        return

    async def round_robin_worker(self, input_requests):
        available_workers = list(self.workers.keys())
        for r in input_requests:
            self.put_req_to_worker(available_workers[self.round_robin_counter], r)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(available_workers)
        return

    async def shortest_queue_worker(self, input_requests):
        # for demonstration purpose only, as waiting for the rpc response on critical path is not efficient
        worker_status = {
            w: await self.get_worker_status(w)
            for w in self.workers
        }
        if any(status is None for status in worker_status.values()):
            # if any worker is not responsive, use local queue length
            worker_status = {
                w: self.workers[w]["request_queue"].qsize()
                for w in self.workers
            }
        else:
            for w, status in worker_status.items():
                if status["idle"]:
                    worker_status[w] = -1
                else:
                    worker_status[w] = status["queue_length"]
                worker_status[w] += self.workers[w]["request_queue"].qsize()

        for r in input_requests:
            worker = min(worker_status, key=worker_status.get)
            self.put_req_to_worker(worker, r)
            worker_status[worker] += 1
        return

    async def shortest_local_queue_worker(self, input_requests):
        for r in input_requests:
            worker = min(
                self.workers,
                key=lambda w: self.workers[w]["request_queue"].qsize())
            self.put_req_to_worker(worker, r)
        return

    def cache_aware_scheduler(self,
                              input_requests,
                              value_func,
                              relax_factor=1,
                              weighted_load=False):
        '''
        A generic cache-aware scheduler that tries to balance the load based on the cache hit rate.
            value_func: a function that takes a request and returns a value indicating the load of the request
            relax_factor: a factor to relax the quota, a larger value could help locality but may cause imbalance
            weighted_load: whether to consider the cache effect on the load
            note: the computation overhead of scheduling might be substantial using Python, consider using C++ later
        '''
        num_reqs = len(input_requests)
        if num_reqs == 0:
            return

        assigned_group = {}
        unassigned_reqs = []
        num_workers = len(self.workers)
        dispatch_count = [0 for _ in range(num_workers)]
        quota = sum(value_func(req)
                    for req in input_requests) / num_workers * relax_factor + 1

        def dispatch_to_worker(req, worker, weighted_factor=1):
            self.put_req_to_worker(worker, req, update_cache=True)
            dispatch_count[worker] += value_func(req) * weighted_factor

        def check_assigned_group(req, matched_request_group):
            if matched_request_group in assigned_group:
                assigned_worker = assigned_group[matched_request_group]
                if dispatch_count[assigned_worker] < quota:
                    dispatch_to_worker(
                        req,
                        w,
                        weighted_factor=RadixCacheHost.GROUPING_THRESHOLD
                        if weighted_load else 1)
                    return True
            return False

        for i, req in enumerate(input_requests):
            matched_workers, matched_request_group = self.radix_cache_host.match_prefix(
                tuple(req.input_ids), request_id=i)
            if len(matched_workers) == 1:
                # greedy assign it to the only matched worker
                w, v = list(matched_workers.items())[0]
                if dispatch_count[w] < quota:
                    dispatch_to_worker(req,
                                       w,
                                       weighted_factor=(1 - v / len(req.input_ids))
                                       if weighted_load else 1)
                    assigned_group[matched_request_group] = w
                else:
                    unassigned_reqs.append((req, matched_request_group))
            elif len(matched_workers) > 1:
                # in case of multiple matched workers, assign it to the one with the most benefit
                if check_assigned_group(req, matched_request_group):
                    continue

                best_worker = -1
                max_benefit = 0
                for w, v in matched_workers.items():
                    # factor benefit by the multiplication of the matched count and remaining quota
                    benefit = v * (quota - dispatch_count[w])
                    if benefit > max_benefit:
                        max_benefit = benefit
                        best_worker = w
                if best_worker != -1:
                    dispatch_to_worker(
                        req,
                        best_worker,
                        weighted_factor=(
                            1 - max_benefit /
                            (quota - dispatch_count[best_worker]) /
                            len(req.input_ids)) if weighted_load else 1)
                    assigned_group[matched_request_group] = best_worker
                else:
                    unassigned_reqs.append((req, matched_request_group))
            else:
                # no worker matched, which is rare and mean the cache is cold
                if check_assigned_group(req, matched_request_group):
                    continue
                else:
                    unassigned_reqs.append((req, matched_request_group))

        while unassigned_reqs:
            req, matched_request_group = unassigned_reqs.pop()
            if check_assigned_group(req, matched_request_group):
                continue
            else:
                leasted_load_worker = random.choice([
                    i for i, x in enumerate(dispatch_count)
                    if x == min(dispatch_count)
                ])
                dispatch_to_worker(req, leasted_load_worker)
                assigned_group[matched_request_group] = leasted_load_worker

    async def cache_aware_req_balance_worker(self, input_requests):

        def value_func(req):
            return 1

        return self.cache_aware_scheduler(input_requests,
                                          value_func)

    async def cache_aware_mem_balance_worker(self, input_requests):

        def value_func(req):
            return len(req.input_ids)

        return self.cache_aware_scheduler(input_requests,
                                          value_func,
                                          weighted_load=True)

    async def pre_scheduled_worker(self, input_requests):
        available_workers = list(self.workers.keys())
        for r in input_requests:
            if r.worker_id and r.worker_id in available_workers:
                worker = r.worker_id
            else:
                # no worker specified or given worker not found, randomly assign
                worker = random.choice(available_workers)
            self.put_req_to_worker(worker, r)
        return

    async def get_worker_status(self, worker_id: int):
        try:
            return await asyncio.wait_for(
                self.workers[worker_id]["client"].get_status(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def remove_dead_workers(self):
        for i, worker in self.workers.items():
            if not worker["client"].liveness:
                self.workers[i]["client"].join()
                # move unsuccessful requests back to the queue
                while not self.workers[i]["request_queue"].empty():
                    self.recv_reqs.append(
                        self.workers[i]["request_queue"].get())
                del self.workers[i]
                logger.info(f"Stale worker {i} removed")

    async def loop_for_forward(self):
        while True:
            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            if next_step_input:
                await self.dispatching(next_step_input)
            # a longer wait time for a more holistic scheduling
            if self.dispatch_method in [
                    DispatchMethod.CACHE_AWARE_REQ_BALANCE,
                    DispatchMethod.CACHE_AWARE_MEM_BALANCE
            ]:
                await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.001)

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            self.recv_reqs.append(recv_req)


def start_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = Controller(server_args.dispatch_method, server_args,
                                port_args)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(controller.loop_for_recv_requests())
    loop.run_until_complete(controller.loop_for_forward())