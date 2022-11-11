# parse command-line args, and return the required object


__all__ = ['parse_optimizer', 'parse_dataset', 'parse_predictor', 'parse_base_model', 'parse_base_head',
           'parse_base_dataset', 'parse_criterion', 'parse_metrics', 'parse_head_dataset',
           'get_base_dataset', 'get_head_dataset', 'get_dataset_name']
from .parse import parse_optimizer, parse_dataset, parse_predictor, parse_base_model, parse_base_head
from .parse import parse_base_dataset, parse_criterion, parse_metrics, parse_head_dataset
from .utils import get_base_dataset, get_head_dataset, get_dataset_name
