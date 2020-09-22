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
