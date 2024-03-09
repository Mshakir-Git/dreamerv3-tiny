import datetime
import collections
import io
import os
import json
import pathlib
import re
import time
import random
from typing import Any

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd
from torch.utils.tensorboard import SummaryWriter

from tinygrad import Tensor,dtypes,TinyJit , nn as tnn

to_np = lambda x: x.detach().cpu().numpy()



def symlog(x):
    return Tensor.sign(x) * Tensor.log(Tensor.abs(x) + 1.0)


def symexp(x):
    return Tensor.sign(x) * (Tensor.exp(Tensor.abs(x)) - 1.0)

class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(requires_grad=True)

    def __exit__(self, *args):
        self._model.requires_grad_(requires_grad=False)


class TimeRecording:
    def __init__(self, comment):
        self._comment = comment

    def __enter__(self):
        self._st = torch.cuda.Event(enable_timing=True)
        self._nd = torch.cuda.Event(enable_timing=True)
        self._st.record()

    def __exit__(self, *args):
        self._nd.record()
        torch.cuda.synchronize()
        print(self._comment, self._st.elapsed_time(self._nd) / 1000)


class Logger:
    def __init__(self, logdir, step):
        self._logdir = logdir
        self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._images = {}
        self._videos = {}
        self.step = step

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        scalars = list(self._scalars.items())
        if fps:
            scalars.append(("fps", self._compute_fps(step)))
        print(f"[{step}]", " / ".join(f"{k} {v:.1f}" for k, v in scalars))
        with (self._logdir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **dict(scalars)}) + "\n")
        for name, value in scalars:
            if "/" not in name:
                self._writer.add_scalar("scalars/" + name, value, step)
            else:
                self._writer.add_scalar(name, value, step)
        for name, value in self._images.items():
            self._writer.add_image(name, value, step)
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if np.issubdtype(value.dtype, np.floating):
                value = np.clip(255 * value, 0, 255).astype(np.uint8)
            B, T, H, W, C = value.shape
            value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
            self._writer.add_video(name, value, step, 16)

        self._writer.flush()
        self._scalars = {}
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration

    def offline_scalar(self, name, value, step):
        self._writer.add_scalar("scalars/" + name, value, step)

    def offline_video(self, name, value, step):
        if np.issubdtype(value.dtype, np.floating):
            value = np.clip(255 * value, 0, 255).astype(np.uint8)
        B, T, H, W, C = value.shape
        value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B * W))
        self._writer.add_video(name, value, step, 16)


def simulate(
    agent,
    envs,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
):
    # initialize or unpack simulation state
    if state is None:
        step, episode = 0, 0
        done = np.ones(len(envs), bool)
        length = np.zeros(len(envs), np.int32)
        obs = [None] * len(envs)
        agent_state = None
        reward = [0] * len(envs)
    else:
        step, episode, done, length, obs, agent_state, reward = state
    while (steps and step < steps) or (episodes and episode < episodes):
        # reset envs if necessary

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            results = [envs[i].reset() for i in indices]
            results = [r() for r in results]
            for index, result in zip(indices, results):
                t = result.copy()
                t = {k: convert(v) for k, v in t.items()}
                # action will be added to transition in add_to_cache
                t["reward"] = 0.0
                t["discount"] = 1.0
                # initial state should be added to cache
                add_to_cache(cache, envs[index].id, t)
                # replace obs with done by initial state
                obs[index] = result
        # step agents
        # print("obs",obs)
        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        # print("obs next",obs)

        action, agent_state = agent(obs, done, agent_state)
        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().numpy()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)
        # step envs
        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done = zip(*[p[:3] for p in results])
        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        episode += int(done.sum())
        length += 1
        step += len(envs)
        length *= 1 - done
        # add to cache
        for a, result, env in zip(action, results, envs):
            o, r, d, info = result
            o = {k: convert(v) for k, v in o.items()}
            transition = o.copy()
            if isinstance(a, dict):
                transition.update(a)
            else:
                transition["action"] = a
            transition["reward"] = r
            transition["discount"] = info.get("discount", np.array(1 - float(d)))
            add_to_cache(cache, env.id, transition)

        if done.any():
            indices = [index for index, d in enumerate(done) if d]
            # logging for done episode
            for i in indices:
                save_episodes(directory, {envs[i].id: cache[envs[i].id]})
                length = len(cache[envs[i].id]["reward"]) - 1
                score = float(np.array(cache[envs[i].id]["reward"]).sum())
                video = cache[envs[i].id]["image"]
                # record logs given from environments
                for key in list(cache[envs[i].id].keys()):
                    if "log_" in key:
                        logger.scalar(
                            key, float(np.array(cache[envs[i].id][key]).sum())
                        )
                        # log items won't be used later
                        cache[envs[i].id].pop(key)

                if not is_eval:
                    step_in_dataset = erase_over_episodes(cache, limit)
                    logger.scalar(f"dataset_size", step_in_dataset)
                    logger.scalar(f"train_return", score)
                    logger.scalar(f"train_length", length)
                    logger.scalar(f"train_episodes", len(cache))
                    logger.write(step=logger.step)
                else:
                    if not "eval_lengths" in locals():
                        eval_lengths = []
                        eval_scores = []
                        eval_done = False
                    # start counting scores for evaluation
                    eval_scores.append(score)
                    eval_lengths.append(length)

                    score = sum(eval_scores) / len(eval_scores)
                    length = sum(eval_lengths) / len(eval_lengths)
                    logger.video(f"eval_policy", np.array(video)[None])

                    if len(eval_scores) >= episodes and not eval_done:
                        logger.scalar(f"eval_return", score)
                        logger.scalar(f"eval_length", length)
                        logger.scalar(f"eval_episodes", len(eval_scores))
                        logger.write(step=logger.step)
                        eval_done = True
    if is_eval:
        # keep only last item for saving memory. this cache is used for video_pred later
        while len(cache) > 1:
            # FIFO
            cache.popitem(last=False)
    return (step - steps, episode - episodes, done, length, obs, agent_state, reward)


def add_to_cache(cache, id, transition):
    if id not in cache:
        cache[id] = dict()
        for key, val in transition.items():
            cache[id][key] = [convert(val)]
    else:
        for key, val in transition.items():
            if key not in cache[id]:
                # fill missing data(action, etc.) at second time
                cache[id][key] = [convert(0 * val)]
                cache[id][key].append(convert(val))
            else:
                cache[id][key].append(convert(val))


def erase_over_episodes(cache, dataset_size):
    step_in_dataset = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
        if (
            not dataset_size
            or step_in_dataset + (len(ep["reward"]) - 1) <= dataset_size
        ):
            step_in_dataset += len(ep["reward"]) - 1
        else:
            del cache[key]
    return step_in_dataset


def convert(value, precision=32):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)


def save_episodes(directory, episodes):
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, episode in episodes.items():
        length = len(episode["reward"])
        filename = directory / f"{filename}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
    return True


def from_generator(generator, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(next(generator))
        data = {}
        for key in batch[0].keys():
            data[key] = []
            for i in range(batch_size):
                data[key].append(batch[i][key])
            data[key] = np.stack(data[key], 0)
        yield data


def sample_episodes(episodes, length, seed=0):
    np_random = np.random.RandomState(seed)
    while True:
        size = 0
        ret = None
        p = np.array(
            [len(next(iter(episode.values()))) for episode in episodes.values()]
        )
        p = p / np.sum(p)
        while size < length:
            episode = np_random.choice(list(episodes.values()), p=p)
            total = len(next(iter(episode.values())))
            # make sure at least one transition included
            if total < 2:
                continue
            if not ret:
                index = int(np_random.randint(0, total - 1))
                ret = {
                    k: v[index : min(index + length, total)].copy()
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][0] = True
            else:
                # 'is_first' comes after 'is_last'
                index = 0
                possible = length - size
                ret = {
                    k: np.append(
                        ret[k], v[index : min(index + possible, total)].copy(), axis=0
                    )
                    for k, v in episode.items()
                    if "log_" not in k
                }
                if "is_first" in ret:
                    ret["is_first"][size] = True
            size = len(next(iter(ret.values())))
        yield ret


def load_episodes(directory, limit=None, reverse=True):
    directory = pathlib.Path(directory).expanduser()
    episodes = collections.OrderedDict()
    total = 0
    if reverse:
        for filename in reversed(sorted(directory.glob("*.npz"))):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            # extract only filename without extension
            episodes[str(os.path.splitext(os.path.basename(filename))[0])] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    else:
        for filename in sorted(directory.glob("*.npz")):
            try:
                with filename.open("rb") as f:
                    episode = np.load(f)
                    episode = {k: episode[k] for k in episode.keys()}
            except Exception as e:
                print(f"Could not load episode: {e}")
                continue
            episodes[str(filename)] = episode
            total += len(episode["reward"]) - 1
            if limit and total >= limit:
                break
    return episodes


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples)
        return torch.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples)
        logprob = self._dist.log_prob(sample)
        return sample[torch.argmax(logprob)][0]

    def entropy(self):
        sample = self._dist.sample(self._samples)
        logprob = self.log_prob(sample)
        return -torch.mean(logprob, 0)



def logsumexp(x, axis=None,dim=1,keepdim=True):
    """Compute the log of the sum of exponentials of input elements."""
    x_max = Tensor.max(x, axis=axis, keepdim=True)
    return Tensor.log(Tensor.sum(Tensor.exp(x - x_max), axis=axis, keepdim=True)) + x_max



from functools import update_wrapper
def logits_to_probs(logits:Tensor):
    return logits.softmax(axis=-1)

def probs_to_logits(probs:Tensor):
    return Tensor.log(probs)

class lazy_property:

    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        # with torch.enable_grad():
        # Tensor.no_grad=False
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


class _lazy_property_and_property(lazy_property, property):
    def __init__(self, wrapped):
        property.__init__(self, wrapped)

class Distribution:
    def __init__(
        self,
        batch_shape = (),
        event_shape = (),
        validate_args = None,
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        if validate_args is not None:
            self._validate_args = validate_args
        super().__init__()

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape


    def _extended_shape(self, sample_shape):
        # if not isinstance(sample_shape, torch.Size):
        #     sample_shape = torch.Size(sample_shape)
        return (sample_shape + self._batch_shape + self._event_shape)
    

class Categorical(Distribution):
   
    def __init__(self, probs=None, logits=None, validate_args=None):

        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.ndim < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            if logits.ndim < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            self.logits = logits - logsumexp(logits,axis=-1, keepdim=True)
            # if(self.logits.shape != tuple((1,6))):
            #     raise KeyboardInterrupt

        self._param:Tensor = self.probs if probs is not None else self.logits
        self._num_events = self._param.shape[-1]
        batch_shape = (
            self._param.shape[:-1] if self._param.ndim > 1 else ()
        )
        self._event_shape = self.param_shape[-1:]
        super().__init__(batch_shape, validate_args=validate_args)



    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)


    @lazy_property
    def logits(self):
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self):
        return logits_to_probs(self.logits)

    @property
    def param_shape(self):
        return self._param.shape

    # @property
    # def mean(self):
    #     return torch.full(
    #         self._extended_shape(),
    #         nan,
    #         dtype=self.probs.dtype,
    #         device=self.probs.device,
    #     )

    @property
    def mode(self):
        return self.probs.argmax(axis=-1)

    # @property
    # def variance(self):
    #     return torch.full(
    #         self._extended_shape(),
    #         nan,
    #         dtype=self.probs.dtype,
    #         device=self.probs.device,
    #     )

    def sample(self, sample_shape):
        # if not isinstance(sample_shape, torch.Size):
        #     sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = Tensor.multinomial(probs_2d, Tensor.zeros(sample_shape).numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, value):
        # if self._validate_args:
        #     self._validate_sample(value)
        value = value.cast(dtypes.int64).unsqueeze(-1)
        value, log_pmf = Tensor._broadcasted(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(value,-1).squeeze(-1)

    def entropy(self):
        # min_real = torch.finfo(self.logits.dtype).min
        # logits = Tensor.clip(self.logits, min_=min_real)
        min_real = 2**((self.logits.dtype.itemsize*8)-1)
        logits = Tensor.clip(self.logits, min_=-min_real ,max_=100000)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    # def enumerate_support(self, expand=True):
    #     num_events = self._num_events
    #     values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
    #     values = values.view((-1,) + (1,) * len(self._batch_shape))
    #     if expand:
    #         values = values.expand((-1,) + self._batch_shape)
    #     return values
    
class OneHotCategorical(Distribution):
  
    def __init__(self, probs=None, logits=None, validate_args=None):
        self._categorical = Categorical(probs, logits)
        self._batch_shape = self._categorical._batch_shape
        self._event_shape = self._categorical.param_shape[-1:]
        # super().__init__(batch_shape, event_shape, validate_args=validate_args)


    # def _new(self, *args, **kwargs):
    #     return self._categorical._new(*args, **kwargs)

    # @property
    # def _param(self):
    #     return self._categorical._param

    @property
    def probs(self):
        return self._categorical.probs

    @property
    def logits(self):
        return self._categorical.logits
    
    @property
    def mean(self):
        return self._categorical.probs

    @property
    def mode(self):
        probs:Tensor = self._categorical.probs
        mode = probs.argmax(axis=-1)
        return one_hot(mode, num_classes=probs.shape[-1]).cast(probs.dtype)

    @property
    def variance(self):
        return self._categorical.probs * (1 - self._categorical.probs)

    @property
    def param_shape(self):
        return self._categorical.param_shape

    def sample(self, sample_shape):
        # sample_shape = (sample_shape)
        probs = self._categorical.probs
        num_events = self._categorical._num_events
        indices = self._categorical.sample(sample_shape)
        return one_hot(indices, num_events).cast(probs.dtype)

    def log_prob(self, value):

        # if self._validate_args:
        #     self._validate_sample(value)
        indices = value.argmax(-1)
        return self._categorical.log_prob(indices)

    def entropy(self):
        return self._categorical.entropy()

    # def enumerate_support(self, expand=True):
    #     n = self.event_shape[0]
    #     values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
    #     values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
    #     if expand:
    #         values = values.expand((n,) + self.batch_shape + (n,))
    #     return values


class OneHotDist(OneHotCategorical):
    def __init__(self, logits:Tensor=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = logits.softmax(axis=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = Tensor.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = one_hot(
            Tensor.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample = sample + (probs - probs.detach())
        return sample
  

class OneHotCategoricalDistribution:
    def __init__(self, probs , logits):
        self.probs = probs
        self._logits = logits - logsumexp(logits,dim=-1, keepdim=True)


    def __call__(self):
        return self
    
    # @property
    # def probs(self):
    #     return self.probs

    @property
    def logits(self):
        return self._logits
    

    def sample(self,shape):
        probs=self.probs.detach()

        cum_sum=probs.cumsum(-1)
        # rand=np.random.random()
        # cum_sum_uni=cum_sum #+ Tensor.uniform(*cum_sum.shape,low=0,high=0.00001) #add noise?
        rand=Tensor.rand((cum_sum.shape[-2],1))
        diff=cum_sum-rand
        # mask=(diff<0).cast(dtypes.float)*2
        # masked_diff=diff+mask
        clipped=diff.clip(min_=0,max_=1)
        swap=clipped.where(0,1)
        swapped_clip=clipped+swap
        min=swapped_clip.min(axis=-1,keepdim=True)
        # min=masked_diff.min(axis=-1,keepdim=True)
        # ret=(masked_diff==min).cast(dtypes.float)
        ret=(swapped_clip==min).cast(dtypes.float)
        # print("Probs",self.probs.numpy())
        # print("Sample",ret.numpy())
        # print("SUM",ret.sum(-1).numpy())

        # assert ret.sum(-1).numpy().all() == Tensor.ones(ret.shape[:-1]).numpy().all()

        #test0
        # min=probs.min(axis=1,keepdim=True)/10000000
        # noise=Tensor.uniform(*(probs.shape),low=-1,high=1)
        # ret:Tensor=probs+(noise*min)
        # max=ret.max(axis=1,keepdim=True)
        # ret=(ret==max).cast(dtypes.float)
        # return ret.detach()
        # ret=Tensor.zeros(probs.shape)
        # if(len(probs.shape[:-1])==2):
        #     x,y=probs.shape[:-1]
        #     for i in range(x):
        #         for j in range(y):
        #             ret[i][j]=self.sample_one_tensor(probs[i][j])
        # elif(len(probs.shape[:-1])==1):
        #     x=probs.shape[:-1][0]
        #     for i in range(x):
        #             ret[i]=self.sample_one_tensor(probs[i])
        # # print(sum(self.probs.numpy()[0][0]),ret[0][0])
        return ret.detach()
    

    def sample_old(self,shape):
        #Improve this
        # print(self.probs,self.logits)
        # probs = Tensor.softmax(self.logits)
        # probs = probs * (1.0 - 0.01) + 0.01 / probs.shape[-1]
        probs=self.probs
        probs_numpy=probs.numpy()
        ret=np.zeros(probs.shape)
        if(len(probs.shape[:-1])==2):
            x,y=probs.shape[:-1]
            for i in range(x):
                for j in range(y):
                    ret[i][j]=self.sample_one(probs_numpy[i][j])
        elif(len(probs.shape[:-1])==1):
            x=probs.shape[:-1][0]
            for i in range(x):
                    ret[i]=self.sample_one(probs_numpy[i])
        # print(sum(self.probs.numpy()[0][0]),ret[0][0])
        return Tensor(ret).cast(dtypes.float)
    
    def rand_sample(self):
        ret=np.zeros(self.probs.shape)
        if(len(self.probs.shape[:-1])==2):
            x,y=self.probs.shape[:-1]
            for i in range(x):
                for j in range(y):
                    ret[i][j]=self.sample_one_random(self.probs[i][j])
        elif(len(self.probs.shape[:-1])==1):
            x=self.probs.shape[:-1][0]
            for i in range(x):
                    ret[i]=self.sample_one_random(self.probs[i])
        # print(sum(self.probs.numpy()[0][0]),ret[0][0])
        return Tensor(ret).cast(dtypes.float)

    def sample_one(self,probs):
        # Generate a random number between 0 and 1
        rand_num = np.random.rand()
        # Accumulate the probabilities to find the category
        cumulative_prob = 0.0
        for i, prob in enumerate(probs):
            cumulative_prob += prob
            if rand_num < cumulative_prob:
                sample = np.zeros(probs.shape)
                sample[i] = 1
                return sample
            
    # def sample_one_tensor(self,probs):
    #     # Generate a random number between 0 and 1
    #     min=float(probs.min().abs().numpy())
    #     noise=Tensor.uniform(*(probs.shape),low=-min/2,high=min/2)
    #     ret:Tensor=probs+noise
    #     max=float(ret.max().numpy())
    #     ret=(ret==max).cast(dtypes.float)
    #     return ret.detach().realize()
            
    def sample_one_random(self,probs):
        # Generate a random number between 0 and 1
        ret=np.zeros(probs.shape[0])
        rand_num = int(np.random.rand()*(probs.shape[0]-0.1))
        # Accumulate the probabilities to find the category
        ret[rand_num]=1
        return ret

def one_hot(cat,classes):
    Tensor.no_grad=True
    ret=Tensor.ones((*cat.shape,classes))
    ret=ret.cumsum(-1)-1
    ret=(ret-cat.unsqueeze(-1))
    ret=Tensor.where(ret==0,1,0)
    Tensor.no_grad=False

    return ret
    # ret=np.zeros((*cat.shape,classes))

    # # for i in range(len(cat.shape)):
    # if(len(ret.shape)==3):
    #     for i in range(cat.shape[0]):
    #         for j in range(cat.shape[1]):
    #             ret[i][j][cat.numpy()[i][j]]=1
    # elif(len(ret.shape)==2):
    #     for i in range(cat.shape[0]):
    #             ret[i][cat.numpy()[i]]=1
    # elif(len(ret.shape)==4):
    #     for i in range(cat.shape[0]):
    #         for j in range(cat.shape[1]):
    #             for k in range(cat.shape[2]):
    #                 ret[i][j][k][cat.numpy()[i][j][k]]=1
    # else:
    #     print("one_hot func len wrong",len(ret.shape))
    #     raise KeyError
    
    # return Tensor(ret,dtype=dtypes.float)
     
GLOBAL_COUNT=0
class OneHotDist_(OneHotCategoricalDistribution):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        ##CHECK NORMALISATION (torch categorical)
        if logits is not None and unimix_ratio > 0.0:
            probs = Tensor.softmax(logits)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = Tensor.log(probs).cast(dtypes.float)
            super().__init__(logits=logits, probs=probs)
        else:
            probs = Tensor.softmax(logits)
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = one_hot(
            Tensor.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        # if(Tensor.argmax(super().logits, axis=-1).shape==(1,32)):
        #     global GLOBAL_COUNT
        #     GLOBAL_COUNT+=1
        # print("big mode in",GLOBAL_COUNT)
        # # print("mode",_mode.numpy())
        # print("super().logits",super().logits.numpy())
        # raise KeyError
        ret=_mode.detach() + self.logits - self.logits.detach()
        return ret

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        # sample = super().sample(sample_shape)
        sample = super().sample(sample_shape)
        probs = self.probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        # print("x.requires_grad",probs.requires_grad)
        # exit()
        prob_grad=probs - probs.detach()
        ret = sample + prob_grad
        return ret


    
    def entropy(self):
        min_real = 2**((self.logits.dtype.itemsize*8)-1)
        logits = Tensor.clip(self.logits, min_=-min_real ,max_=100000)
        p_log_p:Tensor = logits * self.probs
        return -p_log_p.sum(-1).cast(dtypes.float)
    
    def log_prob(self, value_in):
        # raise  KeyError("Meh")
        max=value_in.max(-1)

        # print("log_prob",max.shape[0])
        # exit()
        #Check this in orig
        if(max.shape[0]==1):
            value = value_in.max(-1)[0]
        else:
            value = value_in.max(-1)[1]
        #WRONG use argmax
        value = value.cast(dtypes.float)
        value = value.unsqueeze(-1)
        value, log_pmf = Tensor._broadcasted(value,self.logits)
        value = value[..., :1]
        return log_pmf.gather(value,-1).squeeze(-1)


class DiscDist:
    def __init__(
        self,
        logits,
        low=-20.0,
        high=20.0,
        transfwd=symlog,
        transbwd=symexp,
        device="gpu",
    ):
        self.logits = logits
        self.probs = Tensor.softmax(logits, -1)
        self.buckets = Tensor(np.linspace(low, high, num=255)).cast(dtypes.float)
        self.width = (self.buckets[-1] - self.buckets[0]) / 255
        self.transfwd = transfwd
        self.transbwd = transbwd

    def mean(self):
        _mean = self.probs * self.buckets
        return self.transbwd(Tensor.sum(_mean, axis=-1, keepdim=True))

    def mode(self):
        _mode = self.probs * self.buckets
        ret= self.transbwd(Tensor.sum(_mode, axis=-1, keepdim=True))
        # print("_________mode")
        # print(self.probs[0][0].numpy())
        # print(Tensor.sum(_mode, axis=-1, keepdim=True).numpy())
        # exit()
        return ret

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x):

        # x.requires_grad=False

        x = self.transfwd(x)
        # x(time, batch, 1)

        # below = Tensor.sum((Tensor.clip((((x[..., None] - self.buckets)*1000000000) + 1) ,0,1)).cast(tiny.dtypes.int32), axis=-1) - 1
        # above = self.buckets.shape[0] - Tensor.sum(
        #     (Tensor.clip((self.buckets - x[..., None])*1000000000 ,0,1)).cast(tiny.dtypes.int32), axis=-1
        # )

        below = Tensor.sum((self.buckets <= x[..., None]).cast(dtypes.int32), axis=-1) - 1
        above = self.buckets.shape[0] - Tensor.sum(
            (self.buckets > x[..., None]).cast(dtypes.int32), axis=-1
        )
        # below.requires_grad=False
        # above.requires_grad=False
        # this is implemented using clip at the original repo as the gradients are not backpropagated for the out of limits.
        below = Tensor.clip(below, 0, self.buckets.shape[0] - 1)
        above = Tensor.clip(above, 0, self.buckets.shape[0] - 1)


        # equal = below == above

        # equal = (below - above).cast(dtypes.int)
        equal = (below - above)
        equal=Tensor.where(equal==0,True,False).cast(dtypes.bool)
        # print("Equal",equal.numpy())
        # print("Equal",equal2.numpy())
        # exit()
        dist_to_below = Tensor.where(equal, 1, Tensor.abs(self.buckets[below] - x))
        dist_to_above = Tensor.where(equal, 1, Tensor.abs(self.buckets[above] - x))

        # dist_to_below = Tensor.where(equal, 0, Tensor.abs(self.buckets[below] - x))
        # dist_to_above = Tensor.where(equal, 0, Tensor.abs(self.buckets[above] - x))

        # dist_to_below.requires_grad=False
        # dist_to_above.requires_grad=False

        # zero_or_x_above=(Tensor.abs(self.buckets[above] - x)* Tensor.clip(((Tensor.abs(below - above)*1000000000) + 1),0,1))
        # zero_or_x_below=(Tensor.abs(self.buckets[below] - x)* Tensor.clip(((Tensor.abs(below - above)*1000000000) + 1),0,1))
        # one_or_zero=Tensor.clip( ((Tensor.abs(below - above)*-1000000000)+1),0,1)
       
        # dist_to_above=zero_or_x_above + one_or_zero
        # dist_to_below=zero_or_x_below+ one_or_zero

        # print("dist_to_above",dist_to_above.numpy())
        # print("dist_to_below",dist_to_below.numpy())
        # # print(dist_to_below2.numpy())
        # print("equal",equal.numpy())
        # print("abs below",Tensor.abs(self.buckets[below] - x).numpy())
        # print("abs above",Tensor.abs(self.buckets[above] - x).numpy())
        # print("below",below.numpy())
        # print("above",above.numpy())

        # exit()

        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            one_hot(below, classes=self.buckets.shape[0]) * weight_below[..., None]
            + one_hot(above, classes=self.buckets.shape[0]) * weight_above[..., None]
        )

        log_pred = self.logits - logsumexp(self.logits, -1, keepdim=True)
        target = target.squeeze(-2)

        return (target * log_pred).sum(-1).cast(dtypes.float)

    def log_prob_target(self, target):
        log_pred = super().logits - logsumexp(super().logits, -1, keepdim=True)
        return (target * log_pred).sum(-1)




class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        if(hasattr(loss,"cast")):
            return -loss.cast(dtypes.float)
            
        return -loss

import math
from numbers import Real
def _sum_rightmost(value, dim):
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)

class Normal:
    def __init__(self, shape,mean, std):
            self._mean=mean
            self.std=std
            self._shape=shape

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mean

    def sample(self):
        return Tensor.normal(*self._shape,mean=self._mean, std=self.std)

        
    def entropy(self):
        ret = 0.5 + 0.5 * math.log(2 * math.pi) + Tensor.log(self.std)
        return _sum_rightmost(ret, 1).cast(dtypes.float)


    def log_prob(self, value):
        # compute the variance
        var = self.std**2
        log_scale = (
            math.log(self.std) if isinstance(self.scale, Real) else self.scale.log()
        )
        return (
            -((value - self._mean) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )



class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class Bernoulli_dist:

    def __init__(self,logits):
        self.logits=logits
        self.probs=Tensor.sigmoid(logits) #for binary

    @property
    def mean(self):
        return self.probs

    @property
    def variance(self):
        return self.probs * (1 - self.probs)

    def sample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    


class Bernoulli:
    def __init__(self,logits):
        dist=Bernoulli_dist(logits)
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = Tensor.round(self._dist.mean)
        return _mode.detach() + self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.sample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.logits
        log_probs0 = -Tensor.softplus(_logits)
        log_probs1 = -Tensor.softplus(-_logits)

        return Tensor.sum(log_probs0 * (1 - x) + log_probs1 * x, -1).cast(dtypes.float)


class UnnormalizedHuber(torchd.normal.Normal):
    def __init__(self, loc, scale, threshold=1, **kwargs):
        super().__init__(loc, scale, **kwargs)
        self._threshold = threshold

    def log_prob(self, event):
        return -(
            torch.sqrt((event - self.mean) ** 2 + self._threshold**2)
            - self._threshold
        )

    def mode(self):
        return self.mean


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class TanhBijector(torchd.Transform):
    def __init__(self, validate_args=False, name="tanh"):
        super().__init__()

    def _forward(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        y = torch.where(
            (torch.abs(y) <= 1.0), torch.clamp(y, -0.99999997, 0.99999997), y
        )
        y = torch.atanh(y)
        return y

    def _forward_log_det_jacobian(self, x):
        log2 = torch.math.log(2.0)
        return 2.0 * (log2 - x - torch.softplus(-2.0 * x))



def unbind_dim_0(input:Tensor):
    return tuple(input[i] for i in range(input.shape[0]))
def static_scan_for_lambda_return(fn, inputs, start):
    # print( inputs, start)
    # print("in", inputs[0].numpy(), start.numpy())
    # exit()
    last = start
    indices = range(inputs[0].shape[0])
    indices = reversed(indices)
    flag = True
    for index in indices:
        # (inputs, pcont) -> (inputs[index], pcont[index])
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            outputs = last
            flag = False
        else:
            outputs = Tensor.cat(*[outputs, last], dim=-1)
    outputs = Tensor.reshape(outputs, [outputs.shape[0], outputs.shape[1], 1])
    outputs = Tensor.flip(outputs, [1])
    outputs = unbind_dim_0(outputs)
    return outputs

def lambda_return(reward, value, pcont, bootstrap, lambda_, axis,true_value):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # assert reward.shape.ndims == value.shape.ndims, (reward.shape, value.shape)
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * Tensor.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1 :]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        raise KeyboardInterrupt
        bootstrap = Tensor.zeros_like(value[-1])

    # next_values = Tensor.cat(value[1:], bootstrap[None], dim=0)
    #Hack for nan
    next_values = true_value[1:]
    inputs = reward + pcont * next_values * (1 - lambda_)
    # returns = static_scan(
    #    lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg,
    #    (inputs, pcont), bootstrap, reverse=True)
    # reimplement to optimize performance
    returns = static_scan_for_lambda_return(
        lambda agg, cur0, cur1: cur0 + cur1 * lambda_ * agg, (inputs, pcont), bootstrap
    )
    if axis != 0:
        returns = returns.permute(dims)
    # for i in returns:
    #     i.requires_grad=False
    return returns


class t_Optimizer:
    #NOT USED
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt=tnn.optim.Adam(list(parameters),lr=lr,eps=eps)
        # self._opt = {
        #     "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
        #     "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
        #     "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
        #     "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
        #     "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        # }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.realize().cpu().numpy()
        self._opt.zero_grad()
        # self._scaler.scale(loss).backward(retain_graph=retain_graph)
        # self._scaler.unscale_(self._opt)
        loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        # if self._wd:
        #     self._apply_weight_decay(params)
        # self._scaler.step(self._opt)
        # self._scaler.update()
        self._opt.step()
        self._opt.zero_grad()
        # metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

class Optimizer:
    def __init__(
        self,
        name,
        parameters,
        lr,
        eps=1e-4,
        clip=None,
        wd=None,
        wd_pattern=r".*",
        opt="adam",
        use_amp=False,
    ):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "nadam": lambda: NotImplemented(f"{opt} is not implemented"),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
            "momentum": lambda: torch.optim.SGD(parameters, lr=lr, momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f"{self._name}_loss"] = loss.detach().cpu().numpy()
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        # loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        # self._opt.step()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = self._wd_pattern != r".*"
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

from tinygrad import TinyJit

# @TinyJit
def static_scan_obs_jit(fn, action, embed, is_first,state):
    if(state is not None):
        print(state)
        exit()
    inputs=(action, embed, is_first)
    start=(state,state)
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    outputs={}
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: Tensor(value.lazydata, device=value.device, requires_grad=True).unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: Tensor(value.lazydata, device=value.device, requires_grad=True).unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(Tensor(_last.lazydata, device=_last.device, requires_grad=True).unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = Tensor.cat(
                        *[outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = Tensor.cat(
                                *[outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = Tensor.cat(
                            *[outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]

    post_stoch,post_deter,post_logit,prior_stoch,prior_deter,prior_logit=outputs[0]["stoch"],outputs[0]["deter"],outputs[0]["logit"],outputs[1]["stoch"],outputs[1]["deter"],outputs[1]["logit"]
    return post_stoch.realize(),post_deter.realize(),post_logit.realize(),prior_stoch.realize(),prior_deter.realize(),prior_logit.realize()


def static_scan(fn, inputs, start):
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    outputs={}
    for index in indices:
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if flag:
            if type(last) == type({}):
                outputs = {
                    key: Tensor(value.lazydata, device=value.device, requires_grad=True).unsqueeze(0) for key, value in last.items()
                }
            else:
                outputs = []
                for _last in last:
                    if type(_last) == type({}):
                        outputs.append(
                            {
                                key: Tensor(value.lazydata, device=value.device, requires_grad=True).unsqueeze(0)
                                for key, value in _last.items()
                            }
                        )
                    else:
                        outputs.append(Tensor(_last.lazydata, device=_last.device, requires_grad=True).unsqueeze(0))
            flag = False
        else:
            if type(last) == type({}):
                for key in last.keys():
                    outputs[key] = Tensor.cat(
                        *[outputs[key], last[key].unsqueeze(0)], dim=0
                    )
            else:
                for j in range(len(outputs)):
                    if type(last[j]) == type({}):
                        for key in last[j].keys():
                            outputs[j][key] = Tensor.cat(
                                *[outputs[j][key], last[j][key].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = Tensor.cat(
                            *[outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if type(last) == type({}):
        outputs = [outputs]
    # if not flag:
    #     if type(outputs) == type({}):
    #         for key in outputs.keys():
    #             outputs[key].realize()
    #     else:
    #         for j in range(len(outputs)):
    #             if type(outputs[j]) == type({}):
    #                 for key in outputs[j].keys():
    #                     outputs[j][key].realize()
    #             else:
    #                 outputs[j].realize()
    return outputs


class Every:
    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count


class Once:
    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


class Until:
    def __init__(self, until):
        self._until = until

    def __call__(self, step):
        if not self._until:
            return True
        return step < self._until


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        # print(m)
        # print("torch",m.weight.data.detach().numpy())

        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        # print("torch",m.weight.data.detach().numpy())
        # exit()
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f


def tensorstats(tensor, prefix=None):
    metrics = {
        "mean": to_np(Tensor.mean(tensor)),
        "std": to_np(Tensor.std(tensor)),
        "min": to_np(Tensor.min(tensor)),
        "max": to_np(Tensor.max(tensor)),
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def recursively_collect_optim_state_dict(
    obj, path="", optimizers_state_dicts=None, visited=set()
):
    if optimizers_state_dicts is None:
        optimizers_state_dicts = {}
    # avoid cyclic reference
    if id(obj) in visited:
        return optimizers_state_dicts
    else:
        visited.add(id(obj))
    attrs = obj.__dict__
    if isinstance(obj, torch.nn.Module):
        attrs.update(
            {k: attr for k, attr in obj.named_modules() if "." not in k and obj != attr}
        )
    for name, attr in attrs.items():
        new_path = path + "." + name if path else name
        if isinstance(attr, torch.optim.Optimizer):
            optimizers_state_dicts[new_path] = attr.state_dict()
        elif hasattr(attr, "__dict__"):
            optimizers_state_dicts.update(
                recursively_collect_optim_state_dict(
                    attr, new_path, optimizers_state_dicts, visited
                )
            )
    return optimizers_state_dicts


def recursively_load_optim_state_dict(obj, optimizers_state_dicts):
    for path, state_dict in optimizers_state_dicts.items():
        keys = path.split(".")
        obj_now = obj
        for key in keys:
            obj_now = getattr(obj_now, key)
        obj_now.load_state_dict(state_dict)
