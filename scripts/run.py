import argparse

from _context import vidlu

from vidlu.argument_parsing import parse_datasets_arg, parse_model_arg

# Argument parsing #################################################################################


parser = argparse.ArgumentParser(description='Training script')
subparsers = parser.add_subparsers(dest='cmd')
subparsers.required = True

discr_p = subparsers.add_parser('discr')
discr_p.add_argument('datasets', type=parse_datasets_arg, help=parse_datasets_arg.help)
discr_p.add_argument(
    'model', type=str,
    help='Model configuration, e.g. "resnet(18, head=\'semseg\')", "wrn(28,10)".')
discr_p.add_argument(
    'training', type=str,
    help='Model configuration, e.g. "discriminative(epochs=100, )", "wrn(28,10)".')
discr_p.add_argument('problem', type=str, default="auto",
                     help='Problem configuration, e.g. "auto", "uncertainty".')

args = parser.parse_args()
print(args)
datasets = args.datasets
