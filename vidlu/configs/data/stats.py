cityscapes_mo = dict(mean=[73.15 / 255, 82.9 / 255, 72.3 / 255],
                     std=[47.67 / 255, 48.49 / 255, 47.73 / 255])
cityscapes = dict(mean=[0.28689554, 0.32513303, 0.28389177],
                  std=[0.18696375, 0.19017339, 0.18720214])
# IDD (India Driving Dataset, IDD 20k), computed over a 1200-image sample of the train split.
idd = dict(mean=[0.357792, 0.368317, 0.358331],
           std=[0.274297, 0.283185, 0.29866])
imagenet = dict(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
