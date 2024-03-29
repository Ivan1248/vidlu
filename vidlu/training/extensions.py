from vidlu.utils.func import params, Empty, ArgTree, tree_partial


class TrainerExtension:
    def initialize(self, trainer):
        raise NotImplementedError()

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass


class Stage(TrainerExtension):
    def __init__(self):
        self.stage = 0
        self._handle = None

    def initialize(self, trainer):
        self.stage = 0
        self._handle = trainer.training.completed.add_handler(lambda e: self.increment_stage())

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def increment_stage(self):
        self.stage += 1


class _AdversarialTrainingBase(TrainerExtension):
    def __init__(self, attack_f, eval_attack_f=None):
        self.attack_f, self.eval_attack_f = attack_f, eval_attack_f or ArgTree()
        self.attack, self.eval_attack = None, None

    def initialize(self, trainer):
        def make_attack(a_f):
            return a_f(**(dict(loss=trainer.loss) if params(a_f)['loss'] is Empty else {}))

        self.attack = make_attack(self.attack_f)
        if isinstance(self.eval_attack_f, ArgTree):
            self.eval_attack_f = tree_partial(self.attack_f, self.eval_attack_f)
        self.eval_attack = make_attack(self.eval_attack_f or self.attack_f)


class AdversarialTraining(_AdversarialTrainingBase):
    pass


class SemisupVAT(_AdversarialTrainingBase):
    pass  # intentionally not a subclass of AdversarialTraining because of default metrics etc.
