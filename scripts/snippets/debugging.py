# Swiftnet comparison: run.py/train


def save_activations(state):
    nonlocal e
    fd = debug.FileDict("/tmp/debug.pkl", load_proc=torch.load, save_proc=torch.save)
    fd.clear()
    print(fd.keys())
    from vidlu.modules.elements import with_intermediate_outputs
    x, _ = e.trainer.prepare_batch(state.batch)
    _, activations = with_intermediate_outputs(e.trainer.model, return_dict=True)(x)
    activations = {k: v for k, v in activations.items()}
    activations['input'] = x
    fd.update(activations=activations, params=dict(e.model.named_parameters()))
    exit()


MO = None  # False if args.model.startswith("SwiftNet") else True if args.model == "MoSwiftnetRN18" else None

if MO is True:
    debug.state.check = True
    from functools import partial

    x = debug.FileDict("/tmp/debug.pkl", load_proc=partial(torch.load, map_location=torch.device('cpu')),
                       save_proc=torch.save)['activations']['input']
    e.trainer.model.train()
    e.trainer.model(x.to(e.model.device))
    breakpoint()
if MO is False:
    e.trainer.training.iter_started.add_handler(save_activations)

# end


# Logit warping

# from vidlu.modules.inputwise import TPSWarp
# from vidlu.training.robustness.perturbation import NormalInitializer
# import matplotlib.pyplot as plt
#
# warp = TPSWarp(forward=False).on(state.batch[0])
# with torch.no_grad():
#     NormalInitializer(dict(offsets=(0, 0.1)))(warp, None)
# plt.imsave("/home/igrubisic/images/x.png", state.batch[0][0].permute(1, 2, 0).numpy())

# logits3 = logits[0, :3]
# mi = logits3.min()
# s = logits3.max() - mi
# breakpoint()
# plt.imsave("/home/igrubisic/images/a.png", logits3.permute(1, 2, 0).sub(mi).div(s).cpu().numpy())
# plt.imsave("/home/igrubisic/images/b.png", warp(logits)[0, :3].permute(1, 2, 0).sub(mi).div(s).cpu().numpy())
# logits_p = warp(logits)
# probs = logits.softmax(1)
# probs_p = warp(probs)
# mask = warp(torch.ones(probs[:, :1].shape).to(probs.device)).eq(1.).float()
# return kl_div_l(logits_p, probs_p).max(dim=(-1), keepdim=True)[0].max(-2, keepdim=True)[0] * (mask / mask.mean(dim=(-2, -1), keepdim=True))


# end