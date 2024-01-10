import random
import torch.utils.data
from lib.utils import TensorDict
from lib.data import boundary_modification
import numpy as np
from pytorch_pretrained_bert import BertTokenizer
import cv2
from  torchvision.utils import save_image

def no_processing(data):
    return data


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. 

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_search_frames, num_template_frames=1, processing=no_processing, frame_sample_mode='causal',
                 train_cls=False, pos_prob=0.5,max_seq_len=36, bert_model='bert-base-uncased', bert_path=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            num_template_frames - Number of template frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames
        self.num_template_frames = num_template_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode
        self.max_seq_len = max_seq_len
        bert_path = '/home/data/zgt/CVPR2024/OSTrack-main/pretrained/bert/bert-base-uncased-vocab.txt'
        if bert_path is not None:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None,
                            allow_invisible=False, force_invisible=False):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        if self.train_cls:
            return self.getitem_cls()
        else:
            return self.getitem()

    def extract_token_from_nlp(self, nlp, seq_length):
        """ use tokenizer to convert nlp to tokens
        param:
            nlp:  a sentence of natural language
            seq_length: the max token length, if token length larger than seq_len then cut it,
            elif less than, append '0' token at the reef.
        return:
            token_ids and token_marks
        """
        nlp_token = self.tokenizer.tokenize(nlp)
        if len(nlp_token) > seq_length - 2:
            nlp_token = nlp_token[0:(seq_length - 2)]
        # build tokens and token_ids
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in nlp_token:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        return input_ids, input_mask

    def getitem(self):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)

            if is_video_dataset:
                template_frame_ids = None
                search_frame_ids = None
                gap_increase = 0

                if self.frame_sample_mode == 'causal':
                    # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
                    while search_frame_ids is None:
                        base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
                                                                 max_id=len(visible) - self.num_search_frames)
                        prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
                                                                  min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                                  max_id=base_frame_id[0])
                        if prev_frame_ids is None:
                            gap_increase += 5
                            continue
                        template_frame_ids = base_frame_id + prev_frame_ids
                        search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
                                                                  max_id=template_frame_ids[0] + self.max_gap + gap_increase,
                                                                  num_ids=self.num_search_frames)
                        # Increase gap until a frame is found
                        gap_increase += 5

                elif self.frame_sample_mode == "trident" or self.frame_sample_mode == "trident_pro":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("Illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids, seq_info_dict)
                search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros((H, W))] * self.num_template_frames
                # search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros((H, W))] * self.num_search_frames

                grounding_nl = template_anno['nlp'][0]
                if grounding_nl is not None:
                    grounding_nl_token_ids, grounding_nl_token_masks = self.extract_token_from_nlp(grounding_nl,self.max_seq_len)
                search_masks = [torch.zeros((24, 24))] * self.num_search_frames
                search_seg = [torch.zeros((24, 24))] * self.num_search_frames
                search_gt_mask = [torch.zeros((24, 24))] * self.num_search_frames
                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'search_seg': search_seg,
                                   'search_gt_mask': search_gt_mask,
                                   'nl_token_ids': grounding_nl_token_ids,
                                   'nl_token_masks': grounding_nl_token_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})
                # make data augmentation
                data = self.processing(data)
                a = data['search_anno'][0]
                x, y, w, h = int(torch.clamp(a[0], 0, 1) * 24), int(torch.clamp(a[1], 0, 1) * 24), int(
                    torch.clamp(a[2], 0, 1) * 24), int(torch.clamp(a[3], 0, 1) * 24)
                data['search_gt_mask'][0][:, x: x + w] = 1
                data['search_gt_mask'][0][y: y + h, :] = 1
                data['search_gt_mask'][0][:, 0:x] = 0
                data['search_gt_mask'][0][:, x + w:] = 0
                data['search_gt_mask'][0][0:y, :] = 0
                data['search_gt_mask'][0][y + h:, :] = 0


                # x1 = clamp(random.randint(-2,2) + x,0,24)
                # y1 = clamp(random.randint(-2,2) + y, 0, 24)
                # w1 = clamp(random.randint(-2,2) + w, 0, 24)
                # h1 = clamp(random.randint(-2,2) + h, 0, 24)
                # data['search_seg'][0][:, x1: x1 + w1] = 1
                # data['search_seg'][0][y1: y1 + h1, :] = 1
                # data['search_seg'][0][:, 0:x1] = 0
                # data['search_seg'][0][:, x1 + w1:] = 0
                # data['search_seg'][0][0:y1, :] = 0
                # data['search_seg'][0][y1 + h1:, :] = 0
                #
                # m_x = clamp(abs(x - 12) + random.randint(-12,12),0,24)
                # m_y = clamp(abs(y - 12) + random.randint(-12,12),0,24)
                # m_w = clamp(abs(w - 12) + random.randint(-12,12),0,24)
                # m_h = clamp(abs(h - 12) + random.randint(-12,12),0,24)
                # search_m = [torch.zeros((24, 24))] * self.num_search_frames
                # search_m[0][:, m_x: m_x + m_w] = 1
                # search_m[0][m_y: m_y + m_h,:] = 1
                # search_m[0][:, 0: m_x] = 0
                # search_m[0][:, m_x + m_w:] = 0
                # search_m[0][0: m_y,:] = 0
                # search_m[0][m_y + m_h:,:] = 0
                # data['search_seg'][0] = torch.clamp(data['search_seg'][0] + search_m[0] , 0 ,1)
                # k = data['search_seg'][0]
                # save_image(k, '../../visual/mask.png')


                # check whether data is valid
                valid = data['valid']

                # a = data['search_anno'][0]
                # x , y , w , h = int(torch.clamp(a[0],0,1)*256) , int(torch.clamp(a[1],0,1)*256), int(torch.clamp(a[2],0,1)*256), int(torch.clamp(a[3],0,1)*256)
                # data['search_masks'][0][:,x : x + w] = 255
                # data['search_masks'][0][ y: y + h ,:] = 255
                # data['search_masks'][0][:,0:x] = 0
                # data['search_masks'][0][: ,x+w :] = 0
                # data['search_masks'][0][0:y,:] = 0
                # data['search_masks'][0][y + h:,:] = 0
                # seg = random_modified(gt=data['search_masks'][0].numpy())
                # data['search_seg'] = torch.div(torch.from_numpy(seg).unsqueeze(0),255)


                # data['search_masks'] = torch.div(data['search_masks'], 255)

                # k = data['search_seg'][0]
                # save_image(k, '../../visual/mask.png')
                #
                # k = data['search_gt_mask'][0]
                # save_image(k, '../../visual/seg.png')
                # k = data['search_images'][0]
                # save_image(k, '../../visual/img.png')
                # image = cv2.imread('../../visual/img.png')
                # gt = data['search_anno'][0]
                # x, y, w, h = int(gt[0] * 256), int(gt[1] * 256), int(gt[2] * 256), int(gt[3] * 256)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.imwrite('../../visual/search_box.png', image)

            except:
                valid = False

        return data

    def getitem_cls(self):
        # get data for classification
        """
        args:
            index (int): Index (Ignored since we sample randomly)
            aux (bool): whether the current data is for auxiliary use (e.g. copy-and-paste)

        returns:
            TensorDict - dict containing all the data blocks
        """
        valid = False
        label = None
        while not valid:
            # Select a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]

            is_video_dataset = dataset.is_video_sequence()

            # sample a sequence from the given dataset
            seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
            # sample template and search frame ids
            if is_video_dataset:
                if self.frame_sample_mode in ["trident", "trident_pro"]:
                    template_frame_ids, search_frame_ids = self.get_frame_ids_trident(visible)
                elif self.frame_sample_mode == "stark":
                    template_frame_ids, search_frame_ids = self.get_frame_ids_stark(visible, seq_info_dict["valid"])
                else:
                    raise ValueError("illegal frame sample mode")
            else:
                # In case of image dataset, just repeat the image to generate synthetic video
                template_frame_ids = [1] * self.num_template_frames
                search_frame_ids = [1] * self.num_search_frames
            try:
                # "try" is used to handle trackingnet data failure
                # get images and bounding boxes (for templates)
                template_frames, template_anno, meta_obj_train = dataset.get_frames(seq_id, template_frame_ids,
                                                                                    seq_info_dict)
                H, W, _ = template_frames[0].shape
                template_masks = template_anno['mask'] if 'mask' in template_anno else [torch.zeros(
                    (H, W))] * self.num_template_frames
                # get images and bounding boxes (for searches)
                # positive samples
                if random.random() < self.pos_prob:
                    label = torch.ones(1,)
                    search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames
                # negative samples
                else:
                    label = torch.zeros(1,)
                    if is_video_dataset:
                        search_frame_ids = self._sample_visible_ids(visible, num_ids=1, force_invisible=True)
                        if search_frame_ids is None:
                            search_frames, search_anno, meta_obj_test = self.get_one_search()
                        else:
                            search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids,
                                                                                           seq_info_dict)
                            search_anno["bbox"] = [self.get_center_box(H, W)]
                    else:
                        search_frames, search_anno, meta_obj_test = self.get_one_search()
                    H, W, _ = search_frames[0].shape
                    search_masks = search_anno['mask'] if 'mask' in search_anno else [torch.zeros(
                        (H, W))] * self.num_search_frames

                data = TensorDict({'template_images': template_frames,
                                   'template_anno': template_anno['bbox'],
                                   'template_masks': template_masks,
                                   'search_images': search_frames,
                                   'search_anno': search_anno['bbox'],
                                   'search_masks': search_masks,
                                   'dataset': dataset.get_name(),
                                   'test_class': meta_obj_test.get('object_class_name')})

                # make data augmentation
                data = self.processing(data)
                # add classification label
                data["label"] = label
                # check whether data is valid
                valid = data['valid']
            except:
                valid = False

        return data

    def get_center_box(self, H, W, ratio=1/8):
        cx, cy, w, h = W/2, H/2, W * ratio, H * ratio
        return torch.tensor([int(cx-w/2), int(cy-h/2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_search_frames + self.num_template_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(dataset, is_video_dataset)
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(seq_info_dict["valid"], num_ids=1)
            else:
                search_frame_ids = self._sample_visible_ids(visible, num_ids=1, allow_invisible=True)
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(seq_id, search_frame_ids, seq_info_dict)

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id,
                                                    allow_invisible=True)
                else:
                    f_id = self._sample_visible_ids(visible, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = []
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = []
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(visible, num_ids=1)  # the initial template id
            search_frame_ids = self._sample_visible_ids(visible, num_ids=1)  # the search region id
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(valid, num_ids=1, min_id=min_id, max_id=max_id)
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

def random_modified(gt, iou_max=1.0, iou_min=0.8):
    iou_target = np.random.rand() * (iou_max - iou_min) + iou_min
    seg = boundary_modification.modify_boundary((np.array(gt) > 0.5).astype('uint8') * 255, iou_target=iou_target)
    return seg

def clamp(value , minvalue , maxvalue):
    if value > maxvalue :
        return value
    elif value < minvalue:
        return minvalue
    else:
        return value