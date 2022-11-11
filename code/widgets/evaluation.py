from __future__ import division

import abc
import logging
import time
from collections import OrderedDict

import torch
from torch import nn
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def eval_results(model, loader, *, verbose=False):
    model.eval()

    devices = {name: param.device for name, param in model.named_parameters()}
    devices = set(devices.values())
    if len(devices) == 1:  # all params on the same device
        device = devices.pop()
    else:
        device = devices.pop()
        while device == torch.device('cpu'):
            device = devices.pop()

    with torch.no_grad():
        iterator = tqdm(loader, 'evaluating') if verbose else loader
        data_time = - time.time()
        for it, sample in enumerate(iterator):
            if len(sample) == 3:
                # prompt can be uid in prompt tuning, or stored_inputs for eval_mode='stored'
                inputs, labels, prompt = sample
                assert prompt.device == torch.device('cpu')
                inputs = [inputs, prompt]
            else:
                assert len(sample) == 2, 'len(sample) = {}'.format(len(sample))
                inputs, labels = sample
                inputs = [inputs,]

            data_time += time.time()
            move_time = - time.time()
            # put the kernel inputs to the right device, the prompt will be processed in forward call
            inputs[0] = inputs[0].to(device=device)
            labels = labels.to(device)
            move_time += time.time()
            compute_time = - time.time()
            outputs = model(*inputs)
            compute_time += time.time()
            other_time = - time.time()
            yield outputs, labels
            other_time += time.time()
            multi = 1000
            _logger.debug('[{}/{}] data: {:.3f}ms, move: {:.3f}ms, compute: {:.3f}ms, outside: {:.3f}ms'.format(
                it, len(iterator), data_time * multi, move_time * multi,
                compute_time * multi, other_time * multi
            ))
            data_time = - time.time()


class MetricFunc(abc.ABC):
    @abc.abstractmethod
    def accumulate(self, outputs, labels):
        pass

    @abc.abstractmethod
    def get_result(self):
        pass


class CrossEntropyLoss(MetricFunc):
    def __init__(self):
        self.loss_sum = 0.0
        self.num_samples = 0
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')

    def accumulate(self, outputs, labels):
        self.loss_sum += self.loss_func(outputs, labels).item()
        self.num_samples += outputs.size(0)

    def get_result(self):
        return self.loss_sum / self.num_samples


class BCEWithLogitsLoss(MetricFunc):
    """bce loss for multi-label classification"""
    def __init__(self):
        self.loss_sum = 0.0
        self.num_samples = 0  # num_samples * num_targets
        self.loss_func = nn.BCEWithLogitsLoss(reduction='sum')

    def accumulate(self, outputs, labels):
        self.loss_sum += self.loss_func(outputs, labels).item()
        self.num_samples += outputs.size(0) * outputs.size(1)

    def get_result(self):
        return self.loss_sum / self.num_samples


class Accuracy(MetricFunc):
    def __init__(self):
        self.cnt_correct = 0
        self.num_samples = 0

    def accumulate(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        self.cnt_correct += (predicted == labels).sum().item()
        self.num_samples += outputs.size(0)

    def get_result(self):
        return self.cnt_correct / self.num_samples


class AccuracyPerTask(MetricFunc):
    def __init__(self):
        self.cnt_correct = None
        self.num_samples = 0

    def accumulate(self, outputs, labels):
        predicted = (outputs > 0).to(torch.long)
        if self.cnt_correct is None:
            self.cnt_correct = torch.zeros((predicted.size(1),), dtype=torch.float32, device=outputs.device)
        self.cnt_correct += torch.sum((predicted == labels), axis=0)
        self.num_samples += outputs.size(0)

    def get_result(self):
        result = self.cnt_correct / self.num_samples
        result: torch.Tensor
        result = result.tolist()
        return result


class MultiLabelAccuracy(MetricFunc):
    def __init__(self):
        self.cnt_correct = 0
        self.num_samples = 0

    def accumulate(self, outputs, labels):
        # threshold by 0.5
        predicted = (outputs > 0).to(torch.long)
        self.cnt_correct += (predicted == labels).sum().item()
        self.num_samples += outputs.size(0) * outputs.size(1)

    def get_result(self):
        return self.cnt_correct / self.num_samples


class NumSamples(MetricFunc):
    def __init__(self):
        self.num_samples = 0

    def accumulate(self, outputs, labels):
        self.num_samples += outputs.size(0)

    def get_result(self):
        return self.num_samples


class NumClasses(MetricFunc):
    def __init__(self):
        self.classes = set()

    def accumulate(self, outputs, labels):
        for label in labels:
            self.classes.add(label.item())

    def get_result(self):
        return len(self.classes)


ALL_METRICS = OrderedDict([
    ('cre_loss', CrossEntropyLoss),
    ('accuracy', Accuracy),
    ('num_classes', NumClasses),
    ('num_samples', NumSamples)
])


ALL_METRICS_MULTI_LABEL = OrderedDict([
    ('bce_loss', BCEWithLogitsLoss),
    ('accuracy', MultiLabelAccuracy),
    ('accuracy_per_task', AccuracyPerTask),
    ('num_samples', NumSamples)
])


def eval_loader(model, loader, metrics=None, *, verbose=False):
    if metrics is None:
        metrics = ALL_METRICS

    metric_funcs = OrderedDict([
        (metric_name, metric_func())
        for metric_name, metric_func in metrics.items()
    ])
    model.eval()
    with torch.no_grad():
        for outputs, labels in eval_results(model, loader, verbose=verbose):
            for metric_name, metric_func in metric_funcs.items():
                metric_func.accumulate(outputs, labels)
    results = OrderedDict([
        (metric_name, metric_func.get_result())
        for metric_name, metric_func in metric_funcs.items()
    ])
    return results
