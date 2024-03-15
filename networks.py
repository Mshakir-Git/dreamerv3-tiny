import math
from typing import Any, Union
import numpy as np
import re

import torch
# from torch import nn
# from torch.nn.common_types import _size_2_t
# import torch.nn.functional as F
# from torch import distributions as torchd
# import torch_directml
# dml = torch_directml.device()
from tinygrad import Tensor ,dtypes,TinyJit, nn
import tools
from typing import List, Callable


def prod(l):
    ret=1
    for i in l:
        ret*=i
    return ret



class RSSM():
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        self.num_observe=0
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        act = Tensor.silu
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        self.is_first_bypass=0

        #IMG IN
        inp_layers = []
        if self._discrete:
            inp_dim = self._stoch * self._discrete + num_actions
        else:
            inp_dim = self._stoch + num_actions
        inp_layers.append(Linear_Wrapper(inp_dim, self._hidden, bias=False))
        if norm:
            inp_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        inp_layers.append(act)
        self._img_in_layers = inp_layers #call x.sequential layers
        for layer in  self._img_in_layers: t_weight_init(layer)
        # self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        for layer in self._cell.layers: t_weight_init(layer)
        # self._cell.apply(tools.weight_init)

        #IMG OUT
        img_out_layers = []
        inp_dim = self._deter
        img_out_layers.append(Linear_Wrapper(inp_dim, self._hidden, bias=False))
        if norm:
            img_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        img_out_layers.append(act)
        self._img_out_layers = img_out_layers #call x.sequential layers
        for layer in  self._img_out_layers: t_weight_init(layer)
        # self._img_out_layers.apply(tools.weight_init)

        #OBS OUT
        obs_out_layers = []
        inp_dim = self._deter + self._embed
        obs_out_layers.append(Linear_Wrapper(inp_dim, self._hidden, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(self._hidden, eps=1e-03))
        obs_out_layers.append(act)
        self._obs_out_layers = obs_out_layers #call x.sequential layers
        for layer in  self._obs_out_layers: t_weight_init(layer)
        # self._obs_out_layers.apply(tools.weight_init)

        #DISCRETE STOCH
        if self._discrete:
            self._imgs_stat_layer = Linear_Wrapper(
                self._hidden, self._stoch * self._discrete
            )
            uniform_weight_init(1.0)(self._imgs_stat_layer)
            self._obs_stat_layer = Linear_Wrapper(self._hidden, self._stoch * self._discrete)
            uniform_weight_init(1.0)(self._obs_stat_layer)

        else:
            self._imgs_stat_layer = Linear_Wrapper(self._hidden, 2 * self._stoch)
            uniform_weight_init(1.0)(self._imgs_stat_layer)
            self._obs_stat_layer = Linear_Wrapper(self._hidden, 2 * self._stoch)
            uniform_weight_init(1.0)(self._obs_stat_layer)

        if self._initial == "learned":
            self.W = Tensor(np.zeros((1, self._deter))).cast(dtype=dtypes.float)

    def initial(self, batch_size):
        deter = Tensor.zeros(batch_size, self._deter) #.to(self._device) check org
        if self._discrete:
            state = dict(
                logit=Tensor.zeros([batch_size, self._stoch, self._discrete]),
                stoch=Tensor.zeros([batch_size, self._stoch, self._discrete]),
                deter=deter,
            )
        else:
            state = dict(
                mean=Tensor.zeros([batch_size, self._stoch]),
                std=Tensor.zeros([batch_size, self._stoch]),
                stoch=Tensor.zeros([batch_size, self._stoch]),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"]= Tensor.tanh(self.W).repeat([int(batch_size),1])
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        else:
            raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):

        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        # (batch, time, ch) -> (time, batch, ch)
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        #no jit
        # prev_state[0] means selecting posterior of return(posterior, prior) from obs_step
        # post, prior = tools.static_scan(
        #     lambda prev_state, prev_act, embed, is_first: self.obs_step(
        #         prev_state[0], prev_act, embed, is_first
        #     ),
        #     (action, embed, is_first),
        #     (state, state),
        # )
        
        #both jit (step/scan)
        # def process_obs_return(post_prior):
        #     post_stoch,post_deter,post_logit,prior_stoch,prior_deter,prior_logit=post_prior
        #     _post={"stoch":post_stoch,"deter":post_deter,"logit":post_logit}
        #     _prior={"stoch":prior_stoch,"deter":prior_deter,"logit":prior_logit}
        #     return _post,_prior
        
        # def process_inputs(prev_act,embed,is_first,prev_state):
        #     if(prev_state == None):
        #         return {"prev_act":prev_act.realize(),"embed":embed.realize(),"is_first":is_first.realize(),
        #                 "_stoch":Tensor([2,2,2]).realize(),"deter":Tensor([2,2,2]).realize(),"logit":Tensor([2,2,2]).realize()}
        #     else:
        #         return {"prev_act":prev_act.realize(),"embed":embed.realize(),"is_first":is_first.realize(),
        #                 "_stoch":prev_state["stoch"].realize(),"deter":prev_state["deter"].realize(),"logit":prev_state["logit"].realize()}
        
        # post_stoch,post_deter,post_logit,prior_stoch,prior_deter,prior_logit= tools.static_scan_obs_jit(
        #     lambda prev_state, prev_act, embed, is_first: process_obs_return(self.obs_step_jitted(
        #          **process_inputs(prev_act, embed, is_first,prev_state[0])
        #     )),
        #     action.realize(), embed.realize(), is_first.realize(),state
        # )

        # post={"stoch":post_stoch,"deter":post_deter,"logit":post_logit}
        # prior={"stoch":prior_stoch,"deter":prior_deter,"logit":prior_logit}

        #scan jit
        post_stoch,post_deter,post_logit,prior_stoch,prior_deter,prior_logit= tools.static_scan_obs_jit(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(
                prev_state[0], prev_act, embed, is_first
            ),
            action, embed, is_first,self.initial(is_first[0].shape[0])
        )

        post={"stoch":post_stoch,"deter":post_deter,"logit":post_logit}
        prior={"stoch":prior_stoch,"deter":prior_deter,"logit":prior_logit}
        
        # (batch, time, stoch, discrete_num) -> (batch, time, stoch, discrete_num)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior
    
    def img_step(self, prev_state, prev_action, sample=True):
        # (batch, stoch, discrete_num)
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
            # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
            prev_stoch = prev_stoch.reshape(shape)
        # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
        x = Tensor.cat(*[prev_stoch, prev_action], dim=-1)
        # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
        x = x.sequential(self._img_in_layers) #self._img_in_layers(x)
        for _ in range(self._rec_depth):  # rec depth is not correctly implemented
            deter = prev_state["deter"]
            # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
            x, deter = self._cell(x, [deter])
            deter = deter[0]  # Keras wraps the state in a list.
        # (batch, deter) -> (batch, hidden)
        x = x.sequential(self._img_out_layers) #self._img_out_layers(x)
        # (batch, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("ims", x)
        if sample:
            stoch = self.get_dist(stats).sample()
            # print("img_step sample",stoch.numpy())
        else:
            stoch = self.get_dist(stats).mode()
        prior = {"stoch": stoch, "deter": deter, **stats}
        return prior

    def obs_step(self, prev_state, prev_action, embed, is_first:Tensor, sample=True):
        # print(prev_state == None,sum(is_first.numpy()) == is_first.shape[0],sum(is_first.numpy()) > 0)
        # initialize all prev_state
        # init_mask=(Tensor(is_first.shape[0])-is_first.sum()).cast(dtypes.bool).logical_not()
        # is_first_mask=is_first.sum()
        # print(is_first_mask)
        # exit()
        is_first.realize()
        # prev_action_test=None
        # prev_state_test={}


        is_init_cond=Tensor.sum(is_first)==is_first.shape[0]
        # print(is_init_cond.numpy())
        # exit()
        prev_action_init = Tensor.zeros(*(is_first.shape[0], self._num_actions))

        is_first_cond=(Tensor.sum(is_first) > Tensor([0])).mul(is_init_cond.logical_not())
        # print(is_first_cond.numpy())
        # exit()
        is_first_c = is_first[:, None]
        # print(prev_action.numpy(),is_first_c.numpy())
        prev_action_is_first =prev_action*(1.0 - is_first_c)
        init_state_c = self.initial(is_first_c.shape[0])
        prev_state_c={}
        for key, val in prev_state.items():
            is_first_r = Tensor.reshape(
                is_first_c,
                is_first_c.shape + (1,) * (len(val.shape) - len(is_first_c.shape)),
            )
            prev_state_c[key] = (
                val * (1.0 - is_first_r) + init_state_c[key] * is_first_r
            )

        for key, val in prev_state.items():
            prev_state[key] = prev_state_c[key]*is_first_cond + prev_state[key] * is_first_cond.logical_not()
        
        # print(is_init_cond)
        # print(is_first_cond)
        # print((is_init_cond.logical_and(is_first_cond)).logical_not())
        # exit()
        prev_action=prev_action_init*is_init_cond + prev_action_is_first*is_first_cond + prev_action * (is_init_cond.logical_not() * is_first_cond.logical_not())
        


        # if prev_state == None or sum(is_first.numpy()) == is_first.shape[0]:
        #     # if self.is_first_bypass==0:
        #     # print("Once",is_first)
        #     self.is_first_bypass+=1
        #     prev_state = self.initial(is_first.shape[0])
        #     prev_action = Tensor.zeros(*(is_first.shape[0], self._num_actions)) #.to(self._device)
        #     print(prev_action.numpy())
        # # overwrite the prev_state only where is_first=True
        # elif sum(is_first.numpy()) > 0:
        #     print(self.num_observe)
        #     print(is_first.numpy())
        #     is_first = is_first[:, None]
        #     prev_action *= 1.0 - is_first
        #     # print(prev_action.numpy())
        #     # exit()
        #     init_state = self.initial(is_first.shape[0])
        #     for key, val in prev_state.items():
        #         is_first_r = Tensor.reshape(
        #             is_first,
        #             is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
        #         )
        #         prev_state[key] = (
        #             val * (1.0 - is_first_r) + init_state[key] * is_first_r
        #         )

        # assert np.array_equal(prev_action_test.detach().numpy(),prev_action.detach().numpy())
        # for key, val in prev_state.items():
        #     assert np.array_equal(prev_state_test[key].detach().numpy(),prev_state[key].detach().numpy())
            
        # print(prev_state["stoch"].shape)
        # print(prev_action.shape)
        # exit()
        prior = self.img_step(prev_state, prev_action)
        x = Tensor.cat(*[prior["deter"], embed], dim=-1)
        # (batch_size, prior_deter + embed) -> (batch_size, hidden)
        x = x.sequential(self._obs_out_layers) #self._obs_out_layers(x)
        # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
        stats = self._suff_stats_layer("obs", x)
        if sample:
            dist=self.get_dist(stats)
            stoch = dist.sample()
        else:
            raise KeyboardInterrupt
            stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior
    

    # @TinyJit
    # def obs_step_jitted(self, prev_act, embed, is_first,_stoch,deter,logit,sample=True):

    #     # initialize all prev_state
    #     prev_state={"stoch":_stoch,"deter":deter,"logit":logit}
    #     if len(_stoch.shape) == (3,) or sum(is_first.numpy()) == is_first.shape[0]:
    #         # if self.is_first_bypass==0:
    #         # print("Once")
    #         # exit()
    #         self.is_first_bypass+=1
    #         prev_state = self.initial(is_first.shape[0])
    #         prev_act = Tensor.zeros(*(is_first.shape[0], self._num_actions)) #.to(self._device)
    #     # overwrite the prev_state only where is_first=True
    #     elif sum(is_first.numpy()) > 0:
    #         is_first = is_first[:, None]
    #         prev_act *= 1.0 - is_first
    #         init_state = self.initial(is_first.shape[0])
    #         for key, val in prev_state.items():
    #             is_first_r = Tensor.reshape(
    #                 is_first,
    #                 is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)),
    #             )
    #             prev_state[key] = (
    #                 val * (1.0 - is_first_r) + init_state[key] * is_first_r
    #             )

    #     # prior = self.img_step(prev_state, prev_act)
    #     prev_stoch = prev_state["stoch"]
    #     if self._discrete:
    #         shape = list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete]
    #         # (batch, stoch, discrete_num) -> (batch, stoch * discrete_num)
    #         prev_stoch = prev_stoch.reshape(shape)
    #     # (batch, stoch * discrete_num) -> (batch, stoch * discrete_num + action)
    #     x = Tensor.cat(*[prev_stoch, prev_act], dim=-1)
    #     # (batch, stoch * discrete_num + action, embed) -> (batch, hidden)
    #     x = x.sequential(self._img_in_layers) #self._img_in_layers(x)
    #     for _ in range(self._rec_depth):  # rec depth is not correctly implemented
    #         deter = prev_state["deter"]
    #         # (batch, hidden), (batch, deter) -> (batch, deter), (batch, deter)
    #         x, deter = self._cell(x, [deter])
    #         deter = deter[0]  # Keras wraps the state in a list.
    #     # (batch, deter) -> (batch, hidden)
    #     x = x.sequential(self._img_out_layers) #self._img_out_layers(x)
    #     # (batch, hidden) -> (batch_size, stoch, discrete_num)
    #     stats = self._suff_stats_layer("ims", x)
    #     if sample:
    #         stoch = self.get_dist(stats).sample()
    #     else:
    #         stoch = self.get_dist(stats).mode()
    #     prior = {"stoch": stoch, "deter": deter, **stats}
        
    #     x = Tensor.cat(*[prior["deter"], embed], dim=-1)
    #     # (batch_size, prior_deter + embed) -> (batch_size, hidden)
    #     x = x.sequential(self._obs_out_layers) #self._obs_out_layers(x)
    #     # (batch_size, hidden) -> (batch_size, stoch, discrete_num)
    #     stats = self._suff_stats_layer("obs", x)
    #     if sample:
    #         stoch = self.get_dist(stats).sample()
    #     else:
    #         stoch = self.get_dist(stats).mode()
    #     post = {"stoch": stoch, "deter": prior["deter"], **stats}
    #     post_stoch,post_deter,post_logit=post["stoch"].realize(),post["deter"].realize(),post["logit"].realize()
    #     prior_stoch,prior_deter,prior_logit=prior["stoch"].realize(),prior["deter"].realize(),prior["logit"].realize()

    #     return post_stoch,post_deter,post_logit,prior_stoch,prior_deter,prior_logit

    def get_stoch(self, deter):
        # x = self._img_out_layers(deter)
        x =deter.sequential(self._img_out_layers)
        stats = self._suff_stats_layer("ims", x)
        # print("STATS",stats)
        dist = self.get_dist(stats)
        return dist.mode()
    
    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        assert isinstance(state, dict), state
        action = action
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior
    
    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self._discrete]
            stoch = stoch.reshape(shape)
        return Tensor.cat(*[stoch, state["deter"]], dim=-1)
    
    def get_dist(self, state, dtype=None):
        if self._discrete:
            logit = state["logit"].cast(dtypes.float)
            dist = tools.OneHotDist(logit, unimix_ratio=self._unimix_ratio)
        else:
            print("Not implemented")
            exit()
            mean, std = state["mean"], state["std"]
            dist = tools.ContDist(torchd.normal.Normal(mean, std))
        return dist    

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            if name == "ims":
                x = self._imgs_stat_layer(x)
            elif name == "obs":
                x = self._obs_stat_layer(x)
            else:
                raise NotImplementedError
            mean, std = Tensor.split(x, [self._stoch] * 2, -1)
            mean = {
                "none": lambda: mean,
                "tanh5": lambda: 5.0 * Tensor.tanh(mean / 5.0),
            }[self._mean_act]()
            std = {
                "softplus": lambda: Tensor.softplus(std),
                "abs": lambda: Tensor.abs(std + 1),
                "sigmoid": lambda: Tensor.sigmoid(std),
                "sigmoid2": lambda: 2 * Tensor.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {"mean": mean, "std": std}
        
    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        
        # kld = torchd.kl.kl_divergence
        def _kl_categorical_categorical(a, b):
            return sum(a.probs * Tensor.log(a.probs/b.probs))
        def kld(p, q):
            t:Tensor = p.probs * (p.logits - q.logits)
            # t[(q.probs == 0).expand(t.shape)] = 10000000
            # t[(p.probs == 0).expand(t.shape)] = 0
            return t.sum(-1).sum(-1)
        
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss = value = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
        rep_loss = Tensor.clip(rep_loss, min_=free,max_=100)
        dyn_loss = Tensor.clip(dyn_loss, min_=free,max_=100)
        # print( dyn_scale, dyn_loss.dtype,rep_scale,rep_loss.dtype)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss

class MultiEncoder:
    def __init__(
        self,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        symlog_inputs,
    ):
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)

            self._cnn=ConvEncoder( input_shape, cnn_depth, act, norm, kernel_size, minres)
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum([sum(v) for v in self.mlp_shapes.values()])
            self._mlp = MLP(
                input_size,
                None,
                mlp_layers,
                mlp_units,
                act,
                norm,
                symlog_inputs=symlog_inputs,
                name="Encoder",
            )
            self.outdim += mlp_units

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args,**kwds)
    


    def forward(self, obs):
        # print("***********forward***********")
        outputs = []
        if self.cnn_shapes:
            inputs = Tensor.cat(*[obs[k] for k in self.cnn_shapes], dim=-1)
            outputs.append(self._cnn(inputs))
        # if self.mlp_shapes:
        #     inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
        #     outputs.append(self._mlp(inputs))
        outputs = Tensor.cat(*outputs, dim=-1)
        return outputs






class MultiDecoder():
    def __init__(
        self,
        feat_size,
        shapes,
        mlp_keys,
        cnn_keys,
        act,
        norm,
        cnn_depth,
        kernel_size,
        minres,
        mlp_layers,
        mlp_units,
        cnn_sigmoid,
        image_dist,
        vector_dist,
        outscale,
    ):
        # super(MultiDecoder, self).__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)

        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(
                feat_size,
                shape,
                cnn_depth,
                Tensor.silu,
                norm,
                kernel_size,
                minres,
                outscale=outscale,
                cnn_sigmoid=cnn_sigmoid,
            )
        if self.mlp_shapes:
            self._mlp = MLP(
                feat_size,
                self.mlp_shapes,
                mlp_layers,
                mlp_units,
                act,
                norm,
                vector_dist,
                outscale=outscale,
                name="Decoder",
            )
        self._image_dist = image_dist

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args,**kwds)

    def forward(self, features):
        dists = {}
        if self.cnn_shapes:
            feat = features
            outputs = self._cnn(feat)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = Tensor.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._make_image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "normal":
            #Check pytorch
            raise NotImplementedError
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        raise NotImplementedError(self._image_dist)







def trunc_normal_(tensor:Tensor, mean, std, a, b, generator=None):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    def erfinv(x):
        c1=float(0.8862269254527580)
        c2=float(0.2320136665346544)
        c3=float(0.1275561753055979)
        c4=float(0.0865521292415475)
        x2=x*x
        x3=x*x2
        x5=x3*x2
        x7=x5*x2
        return c1*x + c2*x3 + c3*x5 +c4*x7

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.")

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)  
    # # Uniformly fill tensor with values from [l, u], then translate to
    # # [2l-1, 2u-1].
    # tensor.uniform(2 * l - 1, 2 * u - 1)  
    # # Use inverse cdf transform for normal distribution to get truncated
    # # standard normal
    # tensor=Tensor(torch.erfinv(torch.Tensor(tensor.numpy())).numpy())
    # # tensor.erfinv_()    //Port this
    # # Transform to proper mean, std
    # tensor.mul(std * math.sqrt(2.))
    # tensor.add(mean)   
    # # Clamp to ensure it's in the proper range
    # tensor=Tensor(torch.clamp(torch.Tensor(tensor.numpy()),min=a, max=b).numpy())
    # # tensor.clamp_(min=a, max=b) //Port this
    # ret_tensor=Tensor.uniform(tensor.shape,2 * l - 1, 2 * u - 1)
    Tensor.no_grad=True
    ret_tensor=Tensor.uniform(*tensor.shape,low=2 * l - 1,high= 2 * u - 1)
    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    # tensor.uniform(2 * l - 1, 2 * u - 1, generator=generator)
    # tensor=Tensor(torch.Tensor(tensor.numpy()).uniform_(2 * l - 1, 2 * u - 1).numpy())
    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    # tensor=Tensor(torch.erfinv(torch.Tensor(tensor.numpy())).numpy())
    ret_tensor=erfinv(ret_tensor)
    # Transform to proper mean, std
    ret_tensor=ret_tensor* (std * math.sqrt(2.))
    # tensor=Tensor(torch.Tensor(tensor.numpy()).mul_(std * math.sqrt(2.)).numpy())
    ret_tensor=ret_tensor+mean
    # tensor=Tensor(torch.Tensor(tensor.numpy()).add_(mean).numpy())
    # Clamp to ensure it's in the proper range
    # tensor=Tensor(torch.Tensor(tensor.numpy()).clamp_(min=a, max=b).numpy())
    ret_tensor=ret_tensor.clip(min_=a, max_=b)
    Tensor.no_grad=False
    return ret_tensor.detach()
    
def t_weight_init(m):
    return
    if isinstance(m, Linear_Wrapper):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        m.weight=trunc_normal_(
            m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
        if hasattr(m.bias, "data"):
            m.bias=m.bias.full_like(0.0)
    if isinstance(m, Conv2d_Wrapper) or isinstance(m, ConvTranspose2d_Wrapper):
        # space = m.kernel_size[0] * m.kernel_size[1]
        # in_num = space * m.in_channels
        # out_num = space * m.out_channels
        # denoms = (in_num + out_num) / 2.0
        # scale = 1.0 / denoms
        # std = np.sqrt(scale) / 0.87962566103423978
        # m.weight=trunc_normal_(
        #     m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        # )
        # print("Conv weight",m.weight.numpy())
        # exit()
        if hasattr(m.bias, "data"):
            m.bias=m.bias.full_like(0.0)
    elif isinstance(m, ImgChLayerNorm):
        m.norm.weight=m.norm.weight.full_like(1.0)
        if hasattr(m.norm.bias, "data"):
            m.norm.bias=m.norm.bias.full_like(0.0)


class Conv2d_Wrapper(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        self.in_channels=in_channels
        self.out_channels=out_channels
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


# class Conv2d_Wrapper(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         self.in_channels=in_channels
#         self.out_channels=out_channels
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

#     def calc_same_pad(self, i, k, s, d):
#         return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

#     def __call__(self, x: Tensor):
#         return self.forward(x)
    
#     def forward(self, x:Tensor):
#         ih, iw = x.shape[-2:]
#         pad_h = self.calc_same_pad(
#             i=ih, k=self.kernel_size[0], s=self.stride, d=self.dilation
#         )
#         pad_w = self.calc_same_pad(
#             i=iw, k=self.kernel_size[1], s=self.stride, d=self.dilation
#         )

#         if pad_h > 0 or pad_w > 0:
#             x = x.pad2d(
#                  [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
#             )

#         ret = x.conv2d(
#             self.weight,
#             self.bias,
#             self.groups,
#             self.stride,
#             self.dilation,
#             self.padding,
#         )
#         return ret

class ConvEncoder:
    def __init__(
        self,
        input_shape,
        depth=32,
        act="SiLU",
        norm=True,
        kernel_size=4,
        minres=4,
    ):
        # super(ConvEncoder, self).__init__()
        # act = getattr(torch.nn, act)
        act=Tensor.silu
        h, w, input_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for i in range(stages):
            layers.append(
                Conv2d_Wrapper(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    dilation=0,
                    bias=False,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act)
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2

        self.outdim = out_dim // 2 * h * w
        self.layers: List[Callable[[Tensor], Tensor]] = layers
        for layer in self.layers:
            t_weight_init(layer)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, obs:Tensor):
        obs = obs - 0.5

        # (batch, time, h, w, ch) -> (batch * time, h, w, ch)
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))

        # (batch * time, h, w, ch) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        x = x.sequential(self.layers)
        # (batch * time, ...) -> (batch * time, -1)
        x = x.reshape([x.shape[0], int(prod(x.shape[1:]))]).cast(dtype=dtypes.float)
        # (batch * time, -1) -> (batch, time, -1)
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])

def uniform_weight_init(given_scale):
    def f(m):
        return
        if isinstance(m, Linear_Wrapper):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            # nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            m.weight=Tensor.uniform(*m.weight.shape,low=-limit, high=limit).cast(dtype=dtypes.float).detach()
            
            if hasattr(m.bias, "data"):
                m.bias=m.bias.full_like(0.0)

        elif isinstance(m, Conv2d_Wrapper) or isinstance(m, ConvTranspose2d_Wrapper):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            m.weight=Tensor.uniform(*m.weight.shape,low=-limit, high=limit).cast(dtype=dtypes.float).detach()
            
            if hasattr(m.bias, "data"):
                m.bias=m.bias.full_like(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight=m.weight.full_like(1.0)
            if hasattr(m.bias, "data"):
                m.bias=m.bias.full_like(0.0)

    return f

class Linear_Wrapper(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        self.in_features=in_features
        self.out_features=out_features
        super().__init__(in_features, out_features, bias)


class ConvTranspose2d_Wrapper(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        self.in_channels=in_channels
        self.out_channels=out_channels
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, bias)



class ConvDecoder():
    def __init__(
        self,
        feat_size,
        shape=(3, 64, 64),
        depth=32,
        act=Tensor.silu,
        norm=True,
        kernel_size=4,
        minres=4,
        outscale=1.0,
        cnn_sigmoid=False,
    ):
        # super(ConvDecoder, self).__init__()
        # act = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = int(np.log2(shape[1]) - np.log2(minres))
        self._minres = minres
        out_ch = minres**2 * depth * 2 ** (layer_num - 1)
        self._embed_size = out_ch

        self._linear_layer = Linear_Wrapper(feat_size, out_ch)
        uniform_weight_init(outscale)(self._linear_layer)
        in_dim = out_ch // (minres**2)
        out_dim = in_dim // 2

        layers = []
        h, w = minres, minres
        for i in range(layer_num):
            bias = False
            if i == layer_num - 1:
                out_dim = self._shape[0]
                act = False
                bias = True
                norm = False

            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = self.calc_same_pad(k=kernel_size, s=2, d=1)
            pad_w, outpad_w = self.calc_same_pad(k=kernel_size, s=2, d=1)
            layers.append(
                ConvTranspose2d_Wrapper(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            if act:
                layers.append(act)
            in_dim = out_dim
            out_dim //= 2
            h, w = h * 2, w * 2
        [t_weight_init(m) for m in layers[:-1]]
        uniform_weight_init(outscale)(layers[-1])
        self.layers = layers

    def calc_same_pad(self, k, s, d):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args,**kwds)
        
    def forward(self, features, dtype=None):
        x = self._linear_layer(features)
        # (batch, time, -1) -> (batch * time, h, w, ch)
        x = x.reshape(
            [-1, self._minres, self._minres, self._embed_size // self._minres**2]
        )
        # (batch, time, -1) -> (batch * time, ch, h, w)
        x = x.permute(0, 3, 1, 2)
        # x = self.layers(x)
        x = x.sequential(self.layers)
        # (batch, time, -1) -> (batch, time, ch, h, w)
        mean = x.reshape(features.shape[:-1] + self._shape)
        # (batch, time, ch, h, w) -> (batch, time, h, w, ch)
        mean = mean.permute(0, 1, 3, 4, 2)
        if self._cnn_sigmoid:
            mean = Tensor.sigmoid(mean)
        else:
            mean = mean + 0.5
        return mean



class MLP():
    def __init__(
        self,
        inp_dim,
        shape,
        layers,
        units,
        act=Tensor.silu,
        norm=True,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=1.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        outscale=1.0,
        symlog_inputs=False,
        device="gpu",
        name="NoName",
    ):
        # super(MLP, self).__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        # act = getattr(torch.nn, act)
        self._dist = dist
        self._std = std if isinstance(std, str) else [std]
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs
        self._device = device
        self._name=name
        self.layers = []

        for i in range(layers):
            self.layers.append(
                 Linear_Wrapper(inp_dim, units, bias=False)
            )
            if norm:
                self.layers.append(
                     nn.LayerNorm(units, eps=1e-03)
                )
            self.layers.append(Tensor.silu)
            if i == 0:
                inp_dim = units

        # print(self.layers)
        for layer in self.layers:
            t_weight_init(layer)
        # self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = {}
            for name, shape in self._shape.items():
                self.mean_layer[name] = Linear_Wrapper(inp_dim, int(prod(shape)))
            uniform_weight_init(outscale)(self.mean_layer)
            if isinstance(self._std, str) and self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = {}
                for name, shape in self._shape.items():
                    self.std_layer[name] = Linear_Wrapper(inp_dim, int(prod(shape)))
                uniform_weight_init(outscale)(self.std_layer)
                # self.std_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = Linear_Wrapper(inp_dim, int(prod(self._shape)))
            uniform_weight_init(outscale)(self.mean_layer)
            # self.mean_layer.apply(tools.uniform_weight_init(outscale))
            if isinstance(self._std, str) and self._std == "learned":
                assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
                self.std_layer = Linear_Wrapper(units, int(prod(self._shape)))
                uniform_weight_init(outscale)(self.std_layer)
                # self.std_layer.apply(tools.uniform_weight_init(outscale))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args,**kwds)

    def forward(self, features, dtype=None):
        x = features
        if self._symlog_inputs:
            x = tools.symlog(x)
        out = x.sequential(self.layers)
        #check why out is double
        # Used for encoder output
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            dists = {}
            for name, shape in self._shape.items():
                mean = self.mean_layer[name](out)
                if self._std == "learned":
                    std = self.std_layer[name](out)
                else:
                    std = self._std
                dists.update({name: self.dist(self._dist, mean, std, shape)})
            return dists
        else:
            mean = self.mean_layer(out)
            if isinstance(self._std, str) and self._std == "learned":
                std = self.std_layer(out)
            else:
                if isinstance(self._std, str):
                    std=self._std
                else:
                    std = Tensor(self._std)
            return self.dist(self._dist, mean, std, self._shape)

    def dist(self, dist, mean, std, shape):
        # print("called dist",self._dist,self._name,std.shape)
        if self._dist == "tanh_normal":
            raise NotImplementedError(dist)
        elif self._dist == "normal":
            std = (self._max_std - self._min_std) * Tensor.sigmoid(
                std + 2.0
            ) + self._min_std
            dist = tools.Normal(shape=shape,mean=Tensor.tanh(mean), std=std)
            #check if shape is correct
        elif self._dist == "normal_std_fixed":
            raise NotImplementedError(dist)
        elif self._dist == "trunc_normal":
            raise NotImplementedError(dist)
        elif self._dist == "onehot":
            dist = tools.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_gumble":
            raise NotImplementedError(dist)
        elif dist == "huber":
            raise NotImplementedError(dist)
        elif dist == "binary":
            dist = tools.Bernoulli(logits=mean)
        elif dist == "symlog_disc":
            dist = tools.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = tools.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist






class GRUCell():
    def __init__(self, inp_size, size, norm=True, act=Tensor.tanh, update_bias=-1):
        # super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = []
        self.layers.append( Linear_Wrapper(inp_size + size, 3 * size, bias=False)) #"GRU_linear"
        if norm:
            self.layers.append(nn.LayerNorm(3 * size, eps=1e-03)) #"GRU_norm"

    @property
    def state_size(self):
        return self._size
    
    def __call__(self, inputs, state):
        return self.forward(inputs, state)

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        x=Tensor.cat(*[inputs, state], dim=-1)
        parts = x.sequential(self.layers)  #self.layers(Tensor.cat(*[inputs, state], dim=-1))
        reset, cand, update = Tensor.split(parts, [self._size] * 3, dim=-1)
        reset = Tensor.sigmoid(reset)
        cand = self._act(reset * cand)
        update = Tensor.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]




# class t_Conv2dSamePad(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

#     def calc_same_pad(self, i, k, s, d):
#         return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

#     def convert_pad_shape(self,pad_shape): return tuple(tuple(x) for x in pad_shape)
#     def __call__(self, x: Tensor):
#         return self.forward(x)
#     def forward(self, x):
#         ih, iw = x.shape[-2:]
#         pad_h = self.calc_same_pad(
#             i=ih, k=self.kernel_size[0], s=self.stride, d=self.dilation
#         )
#         pad_w = self.calc_same_pad(
#             i=iw, k=self.kernel_size[1], s=self.stride, d=self.dilation
#         )
#         print("bpad",x.shape)
#         print("pad",((pad_w // 2, pad_w - pad_w // 2),( pad_h // 2, pad_h - pad_h // 2),(0,0),(0,0)))
#         if pad_h > 0 or pad_w > 0:
#             x = Tensor.pad(
#                 x,   ((pad_w // 2, pad_w - pad_w // 2),( pad_h // 2, pad_h - pad_h // 2),(0,0),(0,0))

#             )
#         print("apad",x.shape)

#         print("self.weight",self.weight.shape)
#         # exit()
#         ret = Tensor.conv2d(
#             x,
#             self.weight,
#             self.bias,
#             self.groups,
#             self.stride,
#             0,
#             self.padding,

#         )
#         return ret


    
class ImgChLayerNorm():
    def __init__(self, ch, eps=1e-03):
        # super(ImgChLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(ch, eps=eps)
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x
