import vidlu.modules.inputwise as vmi


def _id_or_index_batch_param(p, keep_grad, idx):
    pi = p
    if isinstance(pi, vmi.BatchParameter):
        pi = p.__getitem__(idx)
        if keep_grad and p.grad is not None:
            pi.grad = p.grad.__getitem__(idx)
    return pi


def _index_batch_params_in_list(list_, keep_grad, idx):
    return [_id_or_index_batch_param(p, keep_grad, idx) for p in list_]


def _index_batch_params_in_dict(dict_, keep_grad, idx):
    return {k: _id_or_index_batch_param(p, keep_grad, idx) for k, p in dict_.items()}


class OptimizerBatchParamsIndexer:
    __slots__ = "optimizer", "keep_grad"

    def __init__(self, optimizer, keep_grad):
        self.optimizer, self.keep_grad = optimizer, keep_grad

    def __getitem__(self, idx):
        param_groups = [
            dict(**pg, params=_index_batch_params_in_list(pg["params"], self.keep_grad, idx))
            for pg in self.optimizer.param_groups]
        return type(self.optimizer)(param_groups, **self.optimizer.defaults)


class ListBatchParamsIndexer:
    __slots__ = "list_", "keep_grad"

    def __init__(self, list_, keep_grad):
        self.list_, self.keep_grad = list_, keep_grad

    def __getitem__(self, idx):
        return _index_batch_params_in_list(self.list_, self.keep_grad, idx)


class DictBatchParamsIndexer:
    __slots__ = "dict_", "keep_grad"

    def __init__(self, list_, keep_grad):
        self.dict_, self.keep_grad = list_, keep_grad

    def __getitem__(self, idx):
        return _index_batch_params_in_dict(self.dict_, self.keep_grad, idx)
