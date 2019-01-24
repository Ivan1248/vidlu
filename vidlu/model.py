import abc
import datetime
import os
from functools import reduce

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..data import DataLoader
from ..ioutils import file, console

from .modeldef import ModelDef


class Model(object):

    def __init__(self,
                 modeldef: ModelDef,
                 training_log_period=1,
                 name="Model"):
        self.name = name

        self.modeldef = modeldef
        self.batch_size = modeldef.training_component.batch_size

        self.training_log_period = training_log_period
        self.log = []
        self.training_step_event_handler = lambda i: \
            console.read_line(impatient=True, discard_non_last=True)

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        with self._graph.as_default():
            self._epoch = tf.Variable(0, False, dtype=tf.int32, name='epoch')
            self._increment_epoch = tf.assign(
                self._epoch, self._epoch + 1, name='increment_epoch')
            self._is_training = tf.placeholder(tf.bool, name='is_training')
            self._mc_dropout = tf.placeholder(
                tf.bool, shape=(), name='mc_dropout')
            self._dropout_active = tf.logical_or(self._is_training,
                                                 self._mc_dropout)

            self.nodes = modeldef.build_graph(
                epoch=self._epoch,
                is_training=self._is_training,
                dropout_active=self._dropout_active)

            self._sess.run(tf.global_variables_initializer())

            param_count = sum(
                reduce((lambda x, y: x * y), var.shape.as_list())
                for var in tf.trainable_variables())
            print(f"Number of parameters: {param_count}")

    def __str__(self):
        return self.name

    def save_state(self, path, save_log=True):
        """
            Saves the trained model as `file_path`.
            If `save_log == True`, `self.log` is saved as `file_path`+'.log'.
        """
        file_path = os.path.join(path, str(self))
        os.makedirs(path, exist_ok=True)
        self._saver.save(self._sess, file_path)
        with open(file_path + ".log", mode='w') as fs:
            fs.write("\n".join(self.log))
            fs.flush()
        print("State saved as '" + file_path + "'.")
        return file_path

    def load_state(self, path):
        self._saver.restore(self._sess, path)
        try:
            self.log = file.read_all_lines(path + ".log")
        except:
            self._log("Log file not found.")
        self._log("State loaded (" + str(self.epoch) + " epochs completed).")

    def load_parameters(self, name_to_value, verbose=True):
        if verbose:
            self._log("Loading parameters...")
        with self._graph.as_default():
            name_to_variable = {v.name: v for v in tf.global_variables()}
            for name, value in (tqdm(name_to_value.items()) if verbose else name_to_value.items()):
                var = name_to_variable[name]
                var.load(value, self._sess)
        if verbose:
            self._log("Parameters loaded...")

    @property
    def epoch(self):
        return self._sess.run(self._epoch)

    def train(self, data: DataLoader, epoch_count=1):

        def train_minibatch(inputs, labels, extra_fetches=[]):
            fetches = [self.nodes['training_step'], self.nodes['loss']] + \
                      list(extra_fetches)
            _, loss, *extra = self._run(
                fetches, inputs, labels, is_training=True)
            if 'training_post_step' in self.nodes:
                self._sess.run(self.nodes['training_post_step'])
            return loss, extra

        def log_part_results(b):
            nonlocal losses
            ev = zip(self.ae_eval_names, self._sess.run(self.ae_evals))
            loss = np.mean(losses)
            self._sess.run(self.ae_reset)
            losses = []
            self._log(f" {self.epoch:3d}.{b:3d}: " +
                      f"{self._eval_str(loss, ev)}")

        losses = []
        self._sess.run(self.ae_reset)
        for _ in range(epoch_count):
            self._sess.run(self._increment_epoch)
            self._log(f"Training: epoch {self.epoch:d} " +
                      f"({len(data)} batches of size {self.batch_size}, " +
                      f"lr={self._sess.run(self.nodes['learning_rate']):.2e})")
            for b, (inputs, labels) in enumerate(data):
                b = b + 1
                loss, _ = train_minibatch(
                    inputs, labels, extra_fetches=[self.ae_accum_batch])
                losses.append(loss)
                if b % self.training_log_period == 0:
                    log_part_results(b)
                end = self.training_step_event_handler(b) == 'q'
            if end:
                return False

    def test(self, data: DataLoader, test_name=None, mc_dropout=False):
        if mc_dropout:
            return self._test_mc_dropout(data, test_name)
        test_name = test_name or "?"
        self._log(f'Testing ({test_name})...')
        loss_sum = 0
        self._sess.run(self.ae_reset)
        for inputs, labels in tqdm(data):
            fetches = [self.nodes['loss'], self.ae_accum_batch]
            loss, _ = self._run(fetches, inputs, labels, is_training=False)
            loss_sum += loss
        loss = loss_sum / len(data)
        ev = zip(self.ae_eval_names, self._sess.run(self.ae_evals))
        self._log(" " + self._eval_str(loss, ev))
        return loss, ev

    def predict(self,
                inputs,
                single_input=False,
                outputs="output",
                mc_dropout=False):
        """
        :inputs: a batch or a single input. If the input is a single example
            outputs will not have a the batch dimension either.
        :param outputs: string list of strings which can be a list like
            ["output", "probs", "logits", "uncertainty"]
        """
        if mc_dropout:
            return self._predict_mc_dropout(inputs, single_input, outputs)
        if single_input:
            inputs = [inputs]
        multiple_output_types = type(outputs) is not str
        if multiple_output_types:
            outputs = self._predict(inputs, outputs)
            if single_input:
                outputs = tuple(r[0] for r in outputs)
        else:
            outputs = self._predict(inputs, [outputs])[0]
            if single_input:
                outputs = outputs[0]
        return outputs

    def _run(self,
             fetches,
             inputs,
             labels=None,
             is_training=None,
             mc_dropout=False):
        feed_dict = {self.nodes['input']: inputs}
        if labels is not None:
            feed_dict[self.nodes['label']] = np.array(labels)
        if self._is_training is not None:
            feed_dict[self._is_training] = is_training
        feed_dict[self._mc_dropout] = mc_dropout
        return self._sess.run(fetches, feed_dict)

    def _predict(self, inputs, outputs):
        fetches = [self.nodes[o] if o in self.nodes else self._graph.get_tensor_by_name(o) for o in
                   outputs]
        return self._run(fetches, inputs, None, is_training=False)

    def _sample_logits_and_probs(self, inputs, sample_count):
        fetches = [self.nodes['logits'], self.nodes['probs']]
        probs_logits_pairs = [
            self._run(fetches, inputs, is_training=False, mc_dropout=True)
            for _ in range(sample_count)
        ]
        return tuple(np.array(x) for x in zip(*probs_logits_pairs))

    def _sample_probs(self, inputs, sample_count):
        fetches = [self.nodes['probs']]
        return np.array([
            self._run(fetches, inputs, is_training=False, mc_dropout=True)[0]
            for _ in range(sample_count)
        ])

    def _test_mc_dropout(self,
                         data: DataLoader,
                         test_name=None,
                         sample_count=50):
        from ..evaluation import NumPyClassificationEvaluator
        ev = None
        test_name = test_name or "?"
        self._log(f'Testing (MC-dropout {sample_count}) ({test_name})...')
        for inputs, labels in data:
            sampled_probs = self._sample_probs(inputs, sample_count)
            probs = sampled_probs.mean(axis=0)
            del sampled_probs
            if ev is None:
                ev = NumPyClassificationEvaluator(class_count=probs.shape[-1])
            pred = np.argmax(probs, axis=-1)
            ev.accumulate(labels, pred)
        evals = ev.evaluate()
        ev.reset()
        self._log(" " + self._eval_str(-1, evals))
        return -1, ev

    def _eval_str(self, loss: float, ev):
        if type(ev) is dict:
            ev = ev.items()
        return f"loss={loss:.4f}, " + ", ".join([f"{k}={v:.4f}" for k, v in ev])

    def _log(self, text: str):
        timestr = datetime.datetime.now().strftime('%H:%M:%S')
        text = f"[{timestr}] {text}"
        self.log.append(text)
        print(text)
