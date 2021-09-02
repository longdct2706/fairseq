# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import numpy as np
from fairseq.data import Dictionary
from fairseq.data.audio.multi_modality_dataset import (
    LangPairMaskDataset,
    ModalityDatasetItem,
    MultiModalityDataset,
)
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from examples.speech_text_joint_to_text.tasks.speech_text_joint import (
    SpeechTextJointToTextTask,
)

from phost.data import (
    S2TAugmentConfig,
    SpeechToTextAugmentDataset,
    SpeechToTextAugmentDatasetCreator,
    S2TJointAugmentConfig,
    SpeechToTextJointAugmentDataset,
    SpeechToTextJointAugmentDatasetCreator,
)

logger = logging.getLogger(__name__)


@register_task("speech_to_text_augment")
class SpeechToTextAugmentTask(SpeechToTextTask):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.data_cfg = S2TAugmentConfig(Path(args.data) / args.config_yaml)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TAugmentConfig(Path(args.data) / args.config_yaml)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextAugmentDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextAugmentDataset(
            split="interactive",
            is_train_split=False,
            cfg=self.data_cfg,
            audio_paths=src_tokens,
            n_frames=src_lengths,
        )

    def begin_epoch(self, epoch, model):
        super().begin_epoch(epoch, model)
        np.random.seed(self.cfg.seed + epoch)
        if epoch == 1:
            return
        for split in list(self.datasets.keys()):
            if split.startswith("train"):
                # Perform a new subsampling at each epoch
                self.load_dataset(split, epoch)


@register_task("speech_text_joint_to_text_augment")
class SpeechTextJointToTextAugmentTask(SpeechTextJointToTextTask):
    def __init__(self, args, src_dict, tgt_dict, infer_tgt_lang_id=None):
        super().__init__(args, src_dict, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)
        self.data_cfg = S2TJointAugmentConfig(Path(args.data) / args.config_yaml)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        data_cfg = S2TJointAugmentConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if (not os.path.isfile(src_dict_path)) or (not os.path.isfile(tgt_dict_path)):
            raise FileNotFoundError("Dict not found: {}".format(args.data))
        src_dict = Dictionary.load(src_dict_path.as_posix())
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())

        print("| src dictionary: {} types".format(len(src_dict)))
        print("| tgt dictionary: {} types".format(len(tgt_dict)))

        if args.parallel_text_data != "":
            if not os.path.isabs(args.parallel_text_data):
                args.parallel_text_data = os.path.join(
                    args.data, args.parallel_text_data
                )

            if args.langpairs is None:
                raise Exception(
                    "Could not infer language pair, please provide it explicitly"
                )
        infer_tgt_lang_id = None
        if args.infer_target_lang != "" and data_cfg.prepend_tgt_lang_tag_no_change:
            tgt_lang_tag = SpeechToTextJointAugmentDataset.LANG_TAG_TEMPLATE.format(
                args.infer_target_lang
            )
            infer_tgt_lang_id = tgt_dict.index(tgt_lang_tag)
            assert infer_tgt_lang_id != tgt_dict.unk()
        return cls(args, src_dict, tgt_dict, infer_tgt_lang_id=infer_tgt_lang_id)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        ast_dataset = SpeechToTextJointAugmentDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            src_dict=None if self.speech_only else self.src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )
        noise_token_id = -1
        text_dataset = None
        if self.args.parallel_text_data != "" and is_train_split:
            text_dataset = self.load_langpair_dataset(
                self.data_cfg.prepend_tgt_lang_tag_no_change,
                1.0,
                epoch=epoch,
            )
            if self.args.mask_text_ratio > 0:
                # add mask
                noise_token_id = (
                    self.src_dict.unk()
                    if self.args.noise_token == ""
                    else self.src_dict.index(self.args.noise_token)
                )
                text_dataset = LangPairMaskDataset(
                    text_dataset,
                    src_bos=self.src_dict.bos(),
                    src_eos=self.src_dict.eos(),
                    noise_id=noise_token_id,
                    mask_ratio=self.args.mask_text_ratio,
                    mask_type=self.args.mask_text_type,
                )

        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "sup_speech",
                    ast_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_positions_text, self.args.max_target_positions),
                    self.args.max_tokens_text
                    if self.args.max_tokens_text is not None
                    else self.args.max_tokens,
                    self.args.batch_size,
                ),
            ]
            ast_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = ast_dataset

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):

        if not isinstance(dataset, MultiModalityDataset):
            return super(SpeechTextJointToTextAugmentTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
            )

        mult_ratio = [self.args.speech_sample_ratio, self.args.text_sample_ratio]
        assert len(dataset.datasets) == 2

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            buffer_size=data_buffer_size,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

    def begin_epoch(self, epoch, model):
        super().begin_epoch(epoch, model)
        np.random.seed(self.cfg.seed + epoch)
        if epoch == 1:
            return
        for split in list(self.datasets.keys()):
            if split.startswith("train"):
                # Perform a new subsampling at each epoch
                self.load_dataset(split, epoch)
