import datetime

from scripts import dirs


def save_trained_model(model,
                       ds_id,
                       net_name,
                       epoch_count,
                       dropout=None,
                       pretrained=None,
                       saved_nets_dir=dirs.SAVED_NETS):
    if dropout:
        net_name += '-do'
    if pretrained:
        net_name += '-pretrained'
    model.save_state(f'{saved_nets_dir}/{ds_id}/' +
                     f'{net_name}-e{epoch_count}/' +
                     f'{datetime.datetime.now():%Y-%m-%d-%H%M}')


def load_trained_model(model,
                       ds_id,
                       net_name,
                       epoch_count,
                       date_code,
                       dropout=None,
                       pretrained=None,
                       saved_nets_dir=dirs.SAVED_NETS):
    if dropout:
        net_name += '-do'
    if pretrained:
        net_name += '-pretrained'
    model.load_state(f'{saved_nets_dir}/{ds_id}/' +
                     f'{net_name}-e{epoch_count}/' + f'{date_code}/Model')
