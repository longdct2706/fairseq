import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform,
    SpeechToTextDatasetItem,
)
from .speech_to_text_augment import (
    S2TAugmentConfig,
    SpeechToTextAugment,
)

logger = logging.getLogger(__name__)


class SpeechToTextAugmentDataset(SpeechToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TAugmentConfig,
        audio_paths: List[str],
        n_frames: List[int],
        **kwargs
    ):
        super().__init__(split, is_train_split, cfg, audio_paths, n_frames, **kwargs)
        self.cfg = cfg
        self.data_augment = SpeechToTextAugment(self.cfg)

    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        source = get_features_or_waveform(
            self.audio_paths[index],
            need_waveform=self.cfg.use_audio_input,
            use_sample_rate=self.cfg.use_sample_rate,
        )
        if self.feature_transforms is not None:
            assert not self.cfg.use_audio_input
            source = self.feature_transforms(source)
        source = torch.from_numpy(source).float()

        # apply effects or keep the original audiowave
        if self.is_train_split and np.random.rand() < self.p_augm:
            source = self.data_augment.augment(source)

        # normalize audiowave
        if self.data_augment.normalize:
            source = self.data_augment.normalize_audio(source)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        return SpeechToTextDatasetItem(index=index, source=source, target=target)


class SpeechToTextAugmentDatasetCreator(SpeechToTextDatasetCreator):
    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TAugmentConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
    ) -> SpeechToTextAugmentDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return SpeechToTextAugmentDataset(
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
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
        )
