def handler(state):
    def _generate_results(results, class_colors, dir="/tmp/semisup", common_prefix="",
                          common_suffix="", method_suffix=""):
        import vidlu.utils.presentation.visualization as vis
        import vidlu.transforms.image as vti

        if common_prefix != "":
            common_prefix = f"_{common_prefix}"
        if common_suffix != "":
            common_suffix = f"_{common_suffix}"
        if method_suffix != "":
            method_suffix = f"_{method_suffix}"
        dir = Path(dir)

        results = {
            k: v.argmax(1) if k.startswith('out') else v
            for k, v in results.items() if k.startswith('x') or k.startswith('out')}
        results = {k: v.detach().cpu() for k, v in results.items()}
        results = {
            k: (vti.chw_to_hwc(v.clamp(0, 1)) if v.shape[1] == 3 else v).numpy()
            for k, v in results.items()}

        colors = vis.normalize_colors(class_colors, insert_zeros=True)
        for k, v in results.items():
            if v.shape[-1] != 3:
                v = vis.colorize_segmentation(v + 1, colors)
            ims = [vti.numpy_to_pil((im * 255).astype(np.uint8)) for im in v]
            for i, im in enumerate(ims):
                path = dir / f"{common_prefix}{i:04}_{k}{common_suffix}.png"
                print(path)
                im.save(path)

        return results

    from vidlu.data.datasets import Cityscapes
    if state.iteration == 0:
        _generate_results(dir='/mnt/sdc1/igrubisic/semisup',
                          class_colors=Cityscapes.info.class_colors, results=state.result,
                          common_prefix=f"{state.epoch:04}_{state.iteration:03}")


def register(trainer):
    trainer.training.iter_completed.add_handler(handler)
