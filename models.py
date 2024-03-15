import copy
import torch
from torch import nn

import networks
import tools

from tinygrad import Tensor,TinyJit ,dtypes, nn as tnn
from tinygrad.features.multi import MultiLazyBuffer
to_np = lambda x: x.detach().cpu().numpy()
import numpy as np

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = Tensor([0.05, 0.95]) #.to(device)

    def __call__(self, x, ema_vals):
        flat_x = Tensor.flatten(x.detach())

        #TEMP
        # x_quantile = torch.quantile(input=flat_x, q=self.range)
        x_quantile = Tensor(np.quantile(a=flat_x.numpy(), q=self.range.numpy()),dtype=dtypes.float)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = Tensor.clip(ema_vals[1] - ema_vals[0], min_=1.0,max_=100000)
        offset = ema_vals[0]

        return offset.cast(dtypes.float).detach(), scale.cast(dtypes.float).detach()





class WorldModel:
    def __init__(self, obs_space, act_space, step, config):
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
      
        self.heads = {}
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name

        # self.t_model_opt = tools.t_Optimizer(
        #     "model",
        #     tnn.state.get_parameters(self),
        #     config.model_lr,
        #     config.opt_eps,
        #     config.grad_clip,
        #     config.weight_decay,
        #     opt=config.opt,
        #     use_amp=self._use_amp,
        # )
        # print(
        #     f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        # )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
        params=[]
        for k,v in tnn.state.get_state_dict(self).items():
            # if(k=="heads.reward._std" or k=="heads.cont._std"):
            #     pass
            # else:
                params.append(v)
        self.opt=tnn.optim.Adam(params,lr=self._config.model_lr,eps=self._config.opt_eps)

    @TinyJit
    def _train_jit(self,**kw_args):

        # data=data.copy()
        # data = self.preprocess(data)
        data=kw_args
        my_metrics = {}

        with Tensor.train():
            embed=self.encoder.forward(data)
            post, prior = self.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            kl_free = self._config.kl_free
            dyn_scale = self._config.dyn_scale
            rep_scale = self._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )

            assert kl_loss.shape == embed.shape[:2], kl_loss.shape
            
            preds = {}
            for name, head in self.heads.items():
                grad_head = name in self._config.grad_heads
                feat = self.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            losses = {}
            for name, pred in preds.items():
                loss = -pred.log_prob(data[name])
                assert loss.shape == embed.shape[:2], (name, loss.shape)
                losses[name] = loss
            scaled = {
                key: value * self._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss
            #custom backward
            m_loss=Tensor.mean(model_loss)
            assert len(m_loss.shape) == 0, m_loss.shape
            my_metrics["my_loss"] = m_loss.numpy()
            # print("World model Loss",my_metrics["my_loss"])



            self.opt.zero_grad()
            m_loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            # metrics = self.t_model_opt(Tensor.mean(model_loss), tnn.state.get_parameters(self))
 
            context = dict(
                    embed=embed,
                    feat=self.dynamics.get_feat(post),
                    kl=kl_value,
                    postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        # ret=(
        # post["stoch"].realize(),post["deter"].realize(),post["logit"].realize(),
        # context["embed"].realize(),context["feat"].realize(),
        # context["kl"].realize(),context["postent"].realize()
        # )
        return post["stoch"].realize(),post["deter"].realize(),post["logit"].realize(),context["embed"].realize(),context["feat"].realize(),context["kl"].realize(),context["postent"].realize(),m_loss.realize(),dyn_loss.realize(),rep_loss.realize()
       
    def _train(self, data):

        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)

        # optimizer
        # params=[]
        # for k,v in tnn.state.get_state_dict(self).items():
        #     if(k=="heads.reward._std" or k=="heads.cont._std"):
        #         pass
        #     else:
        #         params.append(v)
        # for k,v in tnn.state.get_state_dict().items():
        #     if()


        # @TinyJit
        # def _train_step(observe_fn,kl_loss_fn,get_feat_fn,
        #     embed,image,is_terminal,is_first,reward,discount,action,logprob,cont):
        #     data={"image":image,"is_terminal":is_terminal,"is_first":is_first,
        #         "reward":reward,"discount":discount,"action":action,"logprob":logprob,"cont":cont
        #         }
        #     with Tensor.train():
        #         post, prior = observe_fn(
        #             embed, data["action"], data["is_first"]
        #         )
        #         kl_free = self._config.kl_free
        #         dyn_scale = self._config.dyn_scale
        #         rep_scale = self._config.rep_scale
        #         kl_loss, kl_value, dyn_loss, rep_loss = kl_loss_fn(
        #             post, prior, kl_free, dyn_scale, rep_scale
        #         )
        #         assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                
        #         preds = {}
        #         for name, head in self.heads.items():
        #             grad_head = name in self._config.grad_heads
        #             feat = get_feat_fn(post)
        #             feat = feat if grad_head else feat.detach()
        #             pred = head(feat)
        #             if type(pred) is dict:
        #                 preds.update(pred)
        #             else:
        #                 preds[name] = pred
        #         losses = {}
        #         for name, pred in preds.items():
        #             loss = -pred.log_prob(data[name])
        #             assert loss.shape == embed.shape[:2], (name, loss.shape)
        #             losses[name] = loss
        #         scaled = {
        #             key: value * self._scales.get(key, 1.0)
        #             for key, value in losses.items()
        #         }
        #         model_loss = sum(scaled.values()) + kl_loss
        #         #custom backward
        #         m_loss=Tensor.mean(model_loss)
        #         assert len(m_loss.shape) == 0, m_loss.shape
        #         print("World model Loss",m_loss.numpy())
        #         # self.opt.zero_grad()
        #         m_loss.backward()
        #         # self.opt.step()
        #     post_stoch,post_deter,post_logit=post["stoch"],post["deter"],post["logit"]
        #     return m_loss.realize(),embed.realize,kl_value.realize(),post_stoch.realize(),post_deter.realize(),post_logit.realize()

        data=data.copy()
        data = self.preprocess(data)
        my_metrics = {}

        with Tensor.train():
            print("encoder Before",self.encoder._cnn.layers[0].weight)
            embed=self.encoder.forward(data)
            post, prior = self.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            kl_free = self._config.kl_free
            dyn_scale = self._config.dyn_scale
            rep_scale = self._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )

            assert kl_loss.shape == embed.shape[:2], kl_loss.shape
            
            preds = {}
            for name, head in self.heads.items():
                grad_head = name in self._config.grad_heads
                feat = self.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if type(pred) is dict:
                    preds.update(pred)
                else:
                    preds[name] = pred
            losses = {}
            for name, pred in preds.items():
                loss = -pred.log_prob(data[name])
                assert loss.shape == embed.shape[:2], (name, loss.shape)
                losses[name] = loss
            scaled = {
                key: value * self._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            model_loss = sum(scaled.values()) + kl_loss
            #custom backward
            m_loss=Tensor.mean(model_loss)
            assert len(m_loss.shape) == 0, m_loss.shape
            my_metrics["my_loss"] = m_loss.numpy()
            print("World model Loss",my_metrics["my_loss"])



            self.opt.zero_grad()
            m_loss.backward()
            print("encoder After",self.encoder._cnn.layers[0].weight)
            print("_img_in_layers After backward",self.dynamics._img_in_layers[0].weight)
            self.opt.step()
            # self.opt.zero_grad()
            # metrics = self.t_model_opt(Tensor.mean(model_loss), tnn.state.get_parameters(self))
      

        # metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        # metrics["kl_free"] = kl_free
        # metrics["dyn_scale"] = dyn_scale
        # metrics["rep_scale"] = rep_scale
        # metrics["dyn_loss"] = to_np(dyn_loss)
        # metrics["rep_loss"] = to_np(rep_loss)
        # metrics["kl"] = to_np(torch.mean(kl_value))
        # with torch.cuda.amp.autocast(self._use_amp):
            # metrics["prior_ent"] = to_np(
            #     torch.mean(self.dynamics.get_dist(prior).entropy())
            # )
            # metrics["post_ent"] = to_np(
            #     torch.mean(self.dynamics.get_dist(post).entropy())
            # )
        
        # embed=self.encoder.forward(data)
        
        # my_metrics["my_loss"],embed,kl_value,post_stoch,post_deter,post_logit=_train_step(
        #     self.dynamics.observe,self.dynamics.kl_loss,self.dynamics.get_feat,
        #     embed.realize(),
        #    data["image"].realize(),data["is_terminal"].realize(),
        #    data["is_first"].realize(),data["reward"].realize(),data["discount"].realize(),
        #    data["action"].realize(),data["logprob"].realize(),data["cont"].realize() 
        # )
        # post={"stoch":post_stoch,"deter":post_deter,"logit":post_logit}

            context = dict(
                    embed=embed,
                    feat=self.dynamics.get_feat(post),
                    kl=kl_value,
                    postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, {"my_loss":my_metrics["my_loss"]}

    # this function is called during both rollout and training 
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = Tensor(obs["image"]) / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: Tensor(v).cast(dtypes.float) if not isinstance(v, Tensor) else v.cast(dtypes.float) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)
        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        # states, _ = self.dynamics.observe(
        #     embed, data["action"], data["is_first"]
        # )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = Tensor.cat(*[recon[:, :5], openl], dim=1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return Tensor.cat(*[truth, model, error], dim=2)





# def cumprod_dim_0(input:Tensor):
#     out=input.numpy()
#     for i in range(input.shape[0])[1:]:
#         out[i]=out[i]*out[i-1]
#     return Tensor(out)

# def cumprod_dim_0(input:Tensor):
#     out=input.detach()
#     for i in range(input.shape[0])[1:]:
#         out[i]=out[i]*out[i-1]
#     return out.detach()
from tinygrad.helpers import prod

def cumprod(self, axis:int=0, _first_zero=False) -> Tensor:
    return prod(self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0),1)._pool((self.shape[axis],)).split([1 for i in range(self.shape[axis])],dim=-2)).squeeze(2).transpose(axis,-1)

class ImagBehavior:
    def __init__(self, config, world_model):
        # super(ImagBehavior, self).__init__()
        self.test_count=0
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device="cpu",
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)



        self._actor_opt=tnn.optim.Adam(tnn.state.get_parameters(self.actor),lr=config.actor["lr"],eps=config.actor["eps"])
        self._value_opt=tnn.optim.Adam(self.get_value_params(),lr=config.critic["lr"],eps=config.critic["eps"])

        # self._actor_opt = tools.Optimizer(
        #     "actor",
        #     self.actor.parameters(),
        #     config.actor["lr"],
        #     config.actor["eps"],
        #     config.actor["grad_clip"],
        #     **kw,
        # )
        # print(
        #     f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        # )
        # self._value_opt = tools.Optimizer(
        #     "value",
        #     self.value.parameters(),
        #     config.critic["lr"],
        #     config.critic["eps"],
        #     config.critic["grad_clip"],
        #     **kw,
        # )
        # print(
        #     f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        # )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            # self.register_buffer("ema_vals", Tensor.zeros((2,)).to(self._config.device))
            self.ema_vals=Tensor.zeros(2)
            self.reward_ema = RewardEMA(device=self._config.device)

    @TinyJit
    def _train_jit(
        self,
        stoch,
        deter,
        logit
    ):
        #Change this
        self._update_slow_target()
        metrics = {}
        start={"stoch":stoch,"deter":deter,"logit":logit}
        objective = lambda f, s, a: self._world_model.heads["reward"](
            self._world_model.dynamics.get_feat(s)
        ).mode()
        with Tensor.train():
            #ACTOR
            imag_feat, imag_state, imag_action = self._imagine(
                start, self.actor, self._config.imag_horizon
            )
            reward = objective(imag_feat, imag_state, imag_action)
            actor_ent = self.actor(imag_feat).entropy()
            # if self.test_count==7:
            #     # for k,v in imag_state.items():
            #     #     print(k,v.numpy())
            #     exit()
            # self.test_count+=1

            state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
            # this target is not scaled by ema or sym_log.
            target, weights, base = self._compute_target(
                imag_feat, imag_state, reward
            )

            actor_loss, mets = self._compute_actor_loss(
                imag_feat,
                imag_action,
                target,
                weights,
                base,
            )
            # print(actor_ent[:-1, ..., None].shape)

            actor_loss = actor_loss - (self._config.actor["entropy"] * actor_ent[:-1, ..., None])
            actor_loss = Tensor.mean(actor_loss)

            # print("actor_loss",actor_loss.numpy())
            # metrics.update(mets)
            value_input = imag_feat


            #CRITIC (value fn)

            value = self.value(value_input[:-1].detach())
            target = Tensor.stack(target, dim=1)
            # (time, batch, 1), (time, batch, 1) -> (time, batch)
            # print("target",target.shape)
            # target.requires_grad=False
            value_loss = -value.log_prob(target.detach())
            slow_target = self._slow_value(value_input[:-1].detach())
            if self._config.critic["slow_target"]:
                value_loss = value_loss - value.log_prob(slow_target.mode().detach())
            # (time, batch, 1), (time, batch, 1) -> (1,)
            value_loss = Tensor.mean(weights[:-1] * value_loss[:, :, None])
            # print("value_loss",value_loss.numpy())
            # metrics.update(tools.tensorstats_tensor(reward, "imag_reward"))

            # exit()
        # metrics.update(tools.tensorstats(value.mode(), "value"))
        # metrics.update(tools.tensorstats(target, "target"))
        # if self._config.actor["dist"] in ["onehot"]:
        #     metrics.update(
        #         tools.tensorstats_tensor(
        #             Tensor.argmax(imag_action, axis=-1).cast(dtypes.float), "imag_action"
        #         )
        #     )
        # else:
        #     metrics.update(tools.tensorstats_tensor(imag_action, "imag_action"))
        # metrics["actor_entropy"] = to_np(torch.mean(actor_ent))

            #BACKPROP
            self._actor_opt.zero_grad()
            actor_loss.backward()
            self._actor_opt.step()
            self._actor_opt.zero_grad()

            self._value_opt.zero_grad()
            value_loss.backward()
            self._value_opt.step()
            self._value_opt.zero_grad()

            # metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            # metrics.update(self._value_opt(value_loss, self.value.parameters()))

        # metrics_relized=(v.realize() for k,v in metrics.items())
        return actor_loss.realize(),value_loss.realize(),imag_action.realize(),reward.realize(),target.realize()
    



    # def _train(
    #     self,
    #     start,
    #     objective,
    # ):
    #     #Change this
    #     self._update_slow_target()
    #     metrics = {}
    #     with Tensor.train():
    #         #ACTOR
    #         imag_feat, imag_state, imag_action = self._imagine(
    #             start, self.actor, self._config.imag_horizon
    #         )
    #         reward = objective(imag_feat, imag_state, imag_action)
    #         actor_ent = self.actor(imag_feat).entropy()

    #         state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
    #         # this target is not scaled by ema or sym_log.
    #         target, weights, base = self._compute_target(
    #             imag_feat, imag_state, reward
    #         )

    #         actor_loss, mets = self._compute_actor_loss(
    #             imag_feat,
    #             imag_action,
    #             target,
    #             weights,
    #             base,
    #         )
    #         # print(actor_ent[:-1, ..., None].shape)

    #         actor_loss = actor_loss - (self._config.actor["entropy"] * actor_ent[:-1, ..., None])
    #         actor_loss = Tensor.mean(actor_loss)

    #         print("actor_loss",actor_loss.numpy())
    #         metrics.update(mets)
    #         value_input = imag_feat


    #         #CRITIC (value fn)

    #         value = self.value(value_input[:-1].detach())
    #         target = Tensor.stack(target, dim=1)
    #         # (time, batch, 1), (time, batch, 1) -> (time, batch)
    #         # print("target",target.shape)
    #         # target.requires_grad=False
    #         value_loss = -value.log_prob(target.detach())
    #         slow_target = self._slow_value(value_input[:-1].detach())
    #         if self._config.critic["slow_target"]:
    #             value_loss = value_loss - value.log_prob(slow_target.mode().detach())
    #         # (time, batch, 1), (time, batch, 1) -> (1,)
    #         value_loss = Tensor.mean(weights[:-1] * value_loss[:, :, None])
    #         print("value_loss",value_loss.numpy())
    #         metrics.update(tools.tensorstats(reward, "imag_reward"))

    #         # exit()
    #     # metrics.update(tools.tensorstats(value.mode(), "value"))
    #     # metrics.update(tools.tensorstats(target, "target"))
    #     # if self._config.actor["dist"] in ["onehot"]:
    #     #     metrics.update(
    #     #         tools.tensorstats(
    #     #             torch.argmax(imag_action, dim=-1).float(), "imag_action"
    #     #         )
    #     #     )
    #     # else:
    #     #     metrics.update(tools.tensorstats(imag_action, "imag_action"))
    #     # metrics["actor_entropy"] = to_np(torch.mean(actor_ent))

    #         #BACKPROP
    #         self._actor_opt.zero_grad()
    #         actor_loss.backward()
    #         self._actor_opt.step()
    #         self._actor_opt.zero_grad()

    #         self._value_opt.zero_grad()
    #         value_loss.backward()
    #         self._value_opt.step()
    #         self._value_opt.zero_grad()

    #         # metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
    #         # metrics.update(self._value_opt(value_loss, self.value.parameters()))


    #     return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [Tensor.arange(horizon)], (start, None, None)
        )
        # succ, feats, actions = tools.static_scan_imagine(
        #     step, Tensor.arange(horizon).realize(), start
        # )
        states = {k: Tensor.cat(*[start[k][None], v[:-1]], dim=0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            # inp_numpy=np.load("torch_inp_file.np",allow_pickle=True)
            # inp=Tensor(inp_numpy)

            # print(inp.shape)
            # tiny_inp=inp.detach().numpy()
            # tiny_inp_file=open("tiny_inp_file.np","wb+")
            # tiny_inp.dump(tiny_inp_file)
            # tiny_inp_file.close()
            # print(self._world_model.heads["cont"].layers[1].weight.detach().numpy())

            discount = self._config.discount * self._world_model.heads["cont"](inp).mean

            
            # print(repr(inp.numpy()))
            # exit()
        else:
            discount = self._config.discount * Tensor.ones_like(reward)
        
        value = self.value(imag_feat).mode()
        # print("value",value.numpy())
        # print("reward",reward.numpy())
        # print("discount",repr(discount.numpy()))

        
        # reward2.realize()
        # print(reward.numpy())
        
        # if self.test_count==7:
        #     # reward=reward7
        #     # print("value",repr(value.detach().numpy()))
        #     # print("reward",repr(reward.detach().numpy()))
        #     # print("discount",repr(discount.detach().numpy()))
        #     exit()
        # elif self.test_count==8:
        #     exit()
        # self.test_count+=1
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
            true_value=value
        )
        weights = cumprod(
            Tensor.cat(*[Tensor.ones_like(discount[:1]), discount[:-1]])
        ).cast(dtypes.float).detach()
        # print("weights",weights.numpy())
        #e2 reward e1 discount
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = Tensor.stack(target, dim=1)
        # Reward ema
        if self._config.reward_EMA and hasattr(self,"reward_ema"):
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = (normed_target - normed_base).cast(dtypes.float)
            # metrics.update(tools.tensorstats(normed_target, "normed_target"))
            # metrics["EMA_005"] = to_np(self.ema_vals[0])
            # metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action.detach())[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach().cast(dtypes.float)
            )
            #FOR ATARI
            # actor_target.requires_grad=False
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        # if(np.isnan(Tensor.mean(actor_loss).numpy())):
        #     print("-weight",(-weights[:-1]).numpy())
        #     print("actor_target",actor_target.numpy())
    
        # if self.test_count==7:
        #     print("target",target.numpy()) #e4 vs e6
        #     print("-weight",(-weights[:-1]).numpy()) #e3 vs e6
        #     exit()
        # else:
        #     self.test_count+=1
        # print("actor_target",actor_target.numpy()) e4 vs e6
        # print(Tensor.mean(actor_loss).numpy())
        # exit()
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.get_value_params(), self.get_slow_value_params()):
                    Tensor.no_grad=True
                    d = mix * s + (1 - mix) * d
                    Tensor.no_grad=False
            self._updates += 1

    def get_value_params(self):
        params=[]
        for k,v in tnn.state.get_state_dict(self.value).items():
            # if(k=="_std"):
            #     pass
            # else:
                params.append(v)
        return params
    
    def get_slow_value_params(self):
        params=[]
        for k,v in tnn.state.get_state_dict(self._slow_value).items():
            # if(k=="_std"):
            #     pass
            # else:
                params.append(v)
        return params
