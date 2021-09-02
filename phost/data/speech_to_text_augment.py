import logging
from pathlib import Path
from typing import Optional, Union

from augment import EffectChain
import numpy as np
import torch.nn.functional as F
import torch
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.data.audio.speech_to_text_joint_dataset import S2TJointDataConfig

logger = logging.getLogger(__name__)


class S2TAugmentConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def max_source_positions(self) -> int:
        """max number of tokens in the source sequence"""
        return self.config.get("max_source_positions", 6000)

    @property
    def max_target_positions(self) -> int:
        """max number of tokens in the target sequence"""
        return self.config.get("max_target_positions", 1024)

    @property
    def sampling_rate(self) -> int:
        """sampling rate of audio"""
        return self.config.get("sampling_rate", 16000)

    @property
    def echo_gainin(self) -> float:
        """Echo gain in for audio augment"""
        return self.config.get("echo_gainin", 0.8)

    @property
    def echo_gainout(self) -> float:
        """Echo gain out for audio augment"""
        return self.config.get("echo_gainout", 0.9)

    @property
    def sample_ratios(self) -> str:
        """sample ratios of the train subsets"""
        return self.config.get("sample_ratios", "1")

    @property
    def da_p_augm(self) -> str:
        """The probability that data augmentation is applied to an example."""
        return self.config.get("da_p_augm", "1")

    @property
    def da_pitch(self) -> str:
        """The range from which to sample the tempo factor during data augmentation"""
        return self.config.get("da_tempo", "0,0")

    @property
    def da_tempo(self) -> str:
        """The range from which to sample the pitch value during data augmentation.
        Measured in cents (i.e. 100ths of a semitone)"""
        return self.config.get("da_tempo", "1,1")

    @property
    def da_echo_delay(self) -> str:
        """The range from which to sample the echo delay value during data augmentation. \
            Measured in milliseconds"""
        return self.config.get("da_echo_delay", "0,0")

    @property
    def da_echo_decay(self) -> str:
        """The range from which to sample the echo decay factor during data augmentation."""
        return self.config.get("da_echo_decay", "0,0")

    @property
    def normalize(self) -> bool:
        """Whether to normalize the audiowave to zero mean and unit variance."""
        return self.config.get("normalize", True)

    @property
    def interactive_tgt_lang(self) -> Optional[str]:
        """TTarget language to be used with Fairseq's interactive mode."""
        return self.config.get("interactive_tgt_lang", None)


class S2TJointAugmentConfig(S2TJointDataConfig, S2TAugmentConfig):
    def __init__(self, yaml_path: Path):
        super().__init__(yaml_path)


class SpeechToTextAugment(object):
    def __init__(self, cfg: Union[S2TAugmentConfig, S2TJointAugmentConfig]):
        self.cfg = cfg
        self.da_p_augm = self.cfg.da_p_augm
        self.normalize = self.cfg.normalize
        self.max_source_len = min(self.cfg.max_source_positions, self.cfg.max_tokens)

        self.sr = self.cfg.sampling_rate
        self.echo_gainin = self.cfg.echo_gainin
        self.echo_gainout = self.cfg.echo_gainout
        self.info = {"rate": self.sr}

        self.da_effects_info = {
            "tempo": list(map(float, self.cfg.da_tempo.split(","))),
            "pitch": list(map(int, self.cfg.da_pitch.split(","))),
            "echo": {
                "delay": list(map(int, self.cfg.da_echo_delay.split(","))),
                "decay": list(map(float, self.cfg.da_echo_decay.split(","))),
            },
        }
        assert len(set(self.da_effects_info["echo"]["delay"])) == len(
            set(self.da_effects_info["echo"]["decay"])
        ), "Specify ranges for both parameters of echo (delay & decay) or for none"

        # add effects
        self.avai_effects = []
        if self.da_effects_info["tempo"][0] != self.da_effects_info["tempo"][1]:
            self.avai_effects.append("tempo")
        if self.da_effects_info["pitch"][0] != self.da_effects_info["pitch"][1]:
            self.avai_effects.append("pitch")
        cond_delay = (
            self.da_effects_info["echo"]["delay"][0]
            != self.da_effects_info["echo"]["delay"][1]
        )
        cond_decay = (
            self.da_effects_info["echo"]["decay"][0]
            != self.da_effects_info["echo"]["decay"][1]
        )
        if cond_delay and cond_decay:
            self.avai_effects.append("echo")

    def augment(self, input_tensor: torch.tensor) -> torch.tensor:

        src_len = len(input_tensor)

        # init empty chain
        effect_chain = EffectChain()

        if "pitch" in self.avai_effects:

            effect_chain = effect_chain.pitch(
                np.random.randint(*self.da_effects_info["pitch"])
            ).rate(self.sr)

        if "tempo" in self.avai_effects:

            # don't apply a tempo effect that will create an example longer than max_src_len
            min_tempo_value = src_len / self.max_source_len
            sampled_tempo_value = np.random.uniform(
                max(min_tempo_value, self.da_effects_info["tempo"][0]),
                self.da_effects_info["tempo"][1],
            )
            effect_chain = effect_chain.tempo(sampled_tempo_value)

            # adjust to length after tempo
            src_len = int(src_len * sampled_tempo_value)

        if "echo" in self.avai_effects:

            effect_chain = effect_chain.echo(
                self.echo_gainin,
                self.echo_gainout,
                np.random.randint(*self.da_effects_info["echo"]["delay"]),
                np.random.uniform(*self.da_effects_info["echo"]["decay"]),
            )

        # apply effects and reduce channel dimension that is created by default
        # also crop the extra frames that were created by echo delay
        input_tensor_augm = effect_chain.apply(
            input_tensor, src_info=self.info, target_info=self.info
        ).squeeze(0)[:src_len]

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        if torch.isnan(input_tensor_augm).any() or torch.isinf(input_tensor_augm).any():
            return input_tensor
        else:
            return input_tensor_augm

    def normalize_audio(self, input_tensor: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            input_tensor = F.layer_norm(input_tensor, input_tensor.shape)
        return input_tensor

    # def worker_init_fn(worker_id: int) -> None:
    #     np.random.seed(np.random.get_state()[1][0] + worker_id)
