from vidlu.utils.func import partial

import vidlu.modules.elements as vme

norm_f = vme.BatchNorm
act_f = vme.ReLU  # inplace=True can cause bugs, but can be detected with vme.is_modified
conv_f = partial(vme.Conv, padding='half', bias=False)
convt_f = partial(vme.ConvTranspose, bias=False)
