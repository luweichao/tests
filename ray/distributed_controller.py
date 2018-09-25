
from drill.common.task_pool import TaskPool

import ray
import queue
import time
import datetime
import threading
import random


class Controller(object):
    def __init__(self, learner_num, learner_class, learner_conf):
        remote_class = ray.remote(num_cpus=1)(learner_class).remote
        self.learners = []
        for i in range(learner_num):
            actor_id = remote_class(i + 1, **learner_conf)
            self.learners.append(actor_id)
        pass

    def run(self):
        train_step = 0
        while True:
            start = datetime.datetime.now()
            train_step += 1
            results = []
            for learner in self.learners:
                results.append(learner.train.remote())
            for i in range(len(results)):
                print('train_step:%d result:%d %s' % (train_step, i, repr(ray.get(results[i]))))
            end = datetime.datetime.now()
            print('train_step:%d time:%s' % (train_step, repr(end - start)))
            #time.sleep(0.1)


class SyncLearner(object):
    def __init__(self, worker_id, train_batch_size, sampler_class, sampler_conf):
        self.id = worker_id
        self.sampler = sampler_class(worker_id, sampler_conf)
        self.train_batch_size = train_batch_size
        self.step = 0

    def train(self):
        print('worker:%d, step:%d' % (self.id, self.step))
        self.step += 1
        return self.sampler.sample(self.train_batch_size)


class AsyncLearner(object):
    def __init__(self, worker_id, train_batch_size, sampler_class, sampler_conf):
        self.id = worker_id
        self.sampler = sampler_class(worker_id, sampler_conf)
        self.train_batch_size = train_batch_size

        self.queue = queue.Queue(1000)
        self.step = 0

        def training():
            while True:
                print('worker:%d, step:%d' % (self.id, self.step))
                self.step += 1
                batch = self.sampler.sample(self.train_batch_size)
                for item in batch:
                    self.queue.put(item)
        self.thread = threading.Thread(target=training)
        self.thread.setDaemon(True)
        self.thread.start()

    def train(self):
        batch = []
        while len(batch) < self.train_batch_size:
            batch.append(self.queue.get())
        return batch


class Sampler(object):
    def __init__(self, worker_id, sampler_conf):
        self.id = worker_id
        self.sampler_conf = sampler_conf
        self._init()

    def _init(self):
        pass

    def sample(self, batch_size=1):
        raise NotImplementedError

    def _start_actor(self, actor_num, actor_class):
        remote_class = ray.remote(num_cpus=1)(actor_class).remote
        self.actors = []
        for i in range(actor_num):
            actor_id = remote_class(i + 1)
            self.actors.append(actor_id)


class SyncSampler(Sampler):
    def _init(self):
        self._start_actor(**self.sampler_conf)

    def sample(self, batch_size=1):
        sample_tasks = TaskPool()
        for actor in self.actors:
            if sample_tasks.count < batch_size:
                sample_tasks.add(actor, actor.sample.remote())
            else:
                break

        batch = []
        while len(batch) < batch_size:
            task_list = sample_tasks.completed()
            if task_list:
                for actor, sample_id in task_list:
                    try:
                        samples = ray.get(sample_id)
                        batch.extend(samples)
                        if len(batch) >= batch_size:
                            break
                        if len(batch) + sample_tasks.count < batch_size:
                            sample_tasks.add(actor, actor.sample.remote())
                    except Exception as e:
                        print('worker:%d actor:[%s] failed: %s' % (self.id, actor, e))

        return batch


class AsyncSampler(Sampler):
    def _init(self):
        self.sample_queue = queue.Queue(1000)
        self.sample_tasks = TaskPool(50)
        self.sample_thread = None

        self._start_actor(**self.sampler_conf)

        for actor in self.actors:
            self.sample_tasks.add(actor, actor.sample.remote())

        def sampling():
            while True:
                task_list = self.sample_tasks.completed()
                if task_list:
                    for actor, sample_id in task_list:
                        try:
                            samples = ray.get(sample_id)
                            for sample in samples:
                                self.sample_queue.put(sample)
                            self.sample_tasks.add(actor, actor.sample.remote())
                        except Exception as e:
                            print('worker:%d actor[%s] failed: %s' % (self.id, actor, e))

        self.sample_thread = threading.Thread(target=sampling)
        self.sample_thread.setDaemon(True)
        self.sample_thread.start()

    def sample(self, batch_size=1):
        batch = []
        while len(batch) < batch_size:
            batch.append(self.sample_queue.get())
        return batch


def fibonacci(n):
    if n == 0 or n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


class SyncActor(object):
    def __init__(self, actor_id):
        self.id = actor_id
        self.num = self.id * 100000000
        random.seed(actor_id)

    def sample(self, batch_size=1):
        batch = []
        for i in range(batch_size):
            batch.append('%d-%d' % (self.num, fibonacci(random.randint(28, 32))))
            self.num += 1
        return batch


class AsyncActor(object):
    def __init__(self, actor_id):
        self.queue = queue.Queue(100)
        self.id = actor_id
        random.seed(actor_id)

        def acting():
            num = self.id * 100000000
            while True:
                self.queue.put('%d-%d' % (num, fibonacci(random.randint(28, 32))))
                num += 1
        self.thread = threading.Thread(target=acting())
        self.thread.setDaemon(True)
        self.thread.start()

    def sample(self, batch_size=1):
        batch = []
        while len(batch) < batch_size:
            batch.append(self.queue.get())
        return batch


controller_conf = {
    'learner_num': 4,
    'learner_class': AsyncLearner,
    'learner_conf': {
        'train_batch_size': 20,
        'sampler_class': AsyncSampler,
        'sampler_conf': {
            'actor_num': 10,
            'actor_class': SyncActor
        }
    }
}

if __name__ == '__main__':
    ray.init(redis_address='192.168.1.40:6379')
    controller = Controller(**controller_conf)
    controller.run()

