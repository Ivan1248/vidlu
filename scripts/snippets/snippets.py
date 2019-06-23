
print('\n'.join(f"{k} {tuple(v.shape)}" for k, v in trainer.model.state_dict().items()))

visualization.view_predictions(data.train.map(lambda r,trainer=trainer: (trainer.attack.perturb(*trainer.prepare_batch((r.x.reshape((1,)+r.x.shape), r.y.reshape((1,)+r.y.shape)))).squeeze().permute(1, 2, 0).detach().cpu().numpy(), int(r.y.cpu().numpy()))))

visualization.view_predictions(data.train.map(lambda r,trainer=trainer: (trainer.attack.perturb(*trainer.prepare_batch((r.x.reshape((1,)+r.x.shape), r.y.reshape((1,)+r.y.shape)))).squeeze().permute(1, 2, 0).detach().cpu().numpy(), r.y.cpu().numpy())))

# semseg, adversarial
visualization.view_predictions(data.train.map(lambda r,trainer=trainer: (trainer.attack.perturb(*trainer.prepare_batch((r.x.reshape((1,)+r.x.shape), r.y.reshape((1,)+r.y.shape)))).squeeze().permute(1, 2, 0).detach().cpu().numpy(), r.y.cpu().numpy())), infer=lambda x,trainer=trainer: trainer.model(torch.tensor(x).to(device=trainer.model.device).permute(2,0,1).unsqueeze(0)).argmax(1).squeeze().int().cpu().numpy())

visualization.view_predictions(data.train.map(lambda r,trainer=trainer: ((r.x.permute(1, 2, 0).detach().cpu().numpy(), r.y.cpu().numpy()))), infer=lambda x,trainer=trainer: trainer.model(torch.tensor(x).to(device=trainer.model.device).permute(2,0,1).unsqueeze(0)).argmax(1).squeeze().int().cpu().numpy())