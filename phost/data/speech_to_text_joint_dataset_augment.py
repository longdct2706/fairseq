import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from fairseq.data.audio.speech_to_text_joint_dataset import (
    SpeechToTextJointDataset,
    SpeechToTextJointDatasetCreator,
    SpeechToTextJointDatasetItem,
)
from .speech_to_text_augment import (
    S2TJointAugmentConfig,
    SpeechToTextAugment,
)

logger = logging.getLogger(__name__)


class SpeechToTextJointAugmentDataset(SpeechToTextJointDataset):
    def __init__(self, cfg: S2TJointAugmentConfig, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.cfg = cfg
        self.data_augment = SpeechToTextAugment(self.cfg)

    def __getitem__(self, index: int) -> SpeechToTextJointDatasetItem:
        s2t_dataset_item = super().__getitem__(index)

        # apply effects or keep the original audiowave
        if self.is_train_split and np.random.rand() < self.p_augm:
            s2t_dataset_item.source = self.data_augment.augment(s2t_dataset_item.source)

        # normalize audiowave
        if self.data_augment.normalize:
            s2t_dataset_item.source = self.data_augment.normalize_audio(
                s2t_dataset_item.source
            )

        src_tokens = None
        if self.src_texts is not None and self.src_dict is not None:
            src_tokens = self.get_tokenized_src_text(index)
            src_tokens = self.src_dict.encode_line(
                src_tokens, add_if_not_exist=False, append_eos=True
            ).long()
        tgt_lang_tag = None
        if self.cfg.prepend_tgt_lang_tag_no_change:
            # prepend_tgt_lang_tag_no_change: modify prev_output_tokens instead
            tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)

        return SpeechToTextJointDatasetItem(
            index=index,
            source=s2t_dataset_item.source,
            target=s2t_dataset_item.target,
            src_txt_tokens=src_tokens,
            tgt_lang_tag=tgt_lang_tag,
        )


class SpeechToTextJointAugmentDatasetCreator(SpeechToTextJointDatasetCreator):
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TJointAugmentConfig,
        tgt_dict,
        src_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_pre_tokenizer,
        src_bpe_tokenizer,
    ) -> SpeechToTextJointAugmentDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return SpeechToTextJointAugmentDataset(
            split_name=split_name,
            is_train_split=is_train_split,
            cfg=cfg,
            audio_paths=audio_paths,
            n_frames=n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            src_dict=src_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
        )
