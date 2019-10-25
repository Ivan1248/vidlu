from vidlu.utils.func import params, Empty, ArgTree, argtree_partial


class TrainerExtension:
    pass


class AdversarialTraining(TrainerExtension):
    def __init__(self, attack_f, eval_attack_f=None):
        self.attack_f, self.eval_attack_f = attack_f, eval_attack_f or ArgTree()

    def initialize(self, trainer):
        attack_f, eval_attack_f = self.attack_f, self.eval_attack_f
        self.attack = attack_f(
            **(dict(loss=trainer.loss) if params(attack_f)['loss'] is Empty else {}))
        if isinstance(eval_attack_f, ArgTree):
            eval_attack_f = argtree_partial(attack_f, eval_attack_f)
        eval_attack_f = eval_attack_f or attack_f
        self.eval_attack = eval_attack_f(**(
            dict(loss=trainer.loss) if params(eval_attack_f)['loss'] is Empty else {}))


class SemiSupervisedVAT(TrainerExtension):
    __init__ = AdversarialTraining.__init__
    initialize = AdversarialTraining.initialize
