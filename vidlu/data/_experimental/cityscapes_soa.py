class CityscapesFineDatasetExperimentalSOA(Dataset):
    subsets = ['train', 'val', 'test']  # 'test' labels are invalid

    def __init__(self, data_dir, subset='train', downsampling_factor=1, remove_hood=False):
        assert subset in self.__class__.subsets

        assert downsampling_factor >= 1
        from ._cityscapes_labels import labels as cslabels

        self._downsampling_factor = downsampling_factor
        self._shape = np.array([1024, 2048]) // downsampling_factor
        self._remove_hood = remove_hood

        IMG_SUFFIX = "_leftImg8bit.png"
        LAB_SUFFIX = "_gtFine_labelIds.png"
        self._id_to_label = [(l.id, l.trainId) for l in cslabels]

        self._images_dir = Path(f'{data_dir}/left/leftImg8bit/{subset}')
        self._labels_dir = Path(f'{data_dir}/fine_annotations/{subset}')
        self._image_list = [x.relative_to(self._images_dir) for x in self._images_dir.glob('/*/*')]
        self._label_list = [x[:-len(IMG_SUFFIX)] + LAB_SUFFIX for x in self._image_list]

        self.info = {
            'id': 'cityscapes',
            'problem_id': 'semseg',
            'class_count': 19,
            'class_names': [l.name for l in cslabels if l.trainId >= 0],
            'class_colors': [l.color for l in cslabels if l.trainId >= 0],
        }
        if downsampling_factor > 1:
            self.name += f"-downsample_{downsampling_factor}x"
        if remove_hood:
            self.name += f"-remove_hood"

    def get_example(self, idx):
        rh_height = self._shape[0] * 7 // 8

        def load_image():
            img = pimg.open(self._images_dir / self._image_list[idx])
            if self._downsampling_factor > 1:
                img = img.resize(self._shape[::-1], pimg.BILINEAR)
            img = np.array(img, dtype=np.uint8)
            if self._remove_hood:
                img = img[:rh_height, :, :]
            return img

        def load_label():
            lab = pimg.open(self._labels_dir / self._label_list[idx])
            if self._downsampling_factor > 1:
                lab = lab.resize(self._shape[::-1], pimg.NEAREST)
            lab = np.array(lab, dtype=np.int8)
            for id, lb in self._id_to_label:
                lab[lab == id] = lb
            if self._remove_hood:
                lab = lab[:rh_height, :]
            return lab

        return _make_example(x_=load_image, y_=load_label)

    def __len__(self):
        return len(self._image_list)