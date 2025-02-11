# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time

import torch
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.file_utils import logging


class CosyVoice:

    def __init__(self, model_dir, load_jit=True):
        instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.sample_rate = 22050  # 添加采样率作为类属性
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          instruct,
                                          configs['allowed_special'])
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'])
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.fp16.zip'.format(model_dir),
                                '{}/llm.llm.fp16.zip'.format(model_dir))
        del configs

    def list_avaliable_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False):
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False)
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False):
        if self.frontend.instruct is True:
            raise ValueError('{} do not support cross_lingual inference'.format(self.model_dir))
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False):
        if self.frontend.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False)
        for i in self.frontend.text_normalize(tts_text, split=True):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.inference(**model_input, stream=stream):
                speech_len = model_output['tts_speech'].shape[1] / 22050
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False):
        """Voice conversion inference
        Args:
            source_speech_16k: Source audio with 16kHz sampling rate
            prompt_speech_16k: Target voice prompt with 16kHz sampling rate
            stream: Whether to use streaming inference
        Returns:
            Generator yielding converted speech segments
        """
        if self.frontend.instruct is True:
            raise ValueError('{} do not support voice conversion inference'.format(self.model_dir))

        # 添加输入验证
        if source_speech_16k is None or prompt_speech_16k is None:
            raise ValueError("Source speech and prompt speech cannot be None")

        if not isinstance(source_speech_16k, torch.Tensor) or not isinstance(prompt_speech_16k, torch.Tensor):
            raise ValueError("Inputs must be torch tensors")

        if source_speech_16k.numel() == 0 or prompt_speech_16k.numel() == 0:
            raise ValueError("Input tensors cannot be empty")

        model_input = self.frontend.frontend_cross_lingual("", prompt_speech_16k)
        model_input['source_speech'] = source_speech_16k

        start_time = time.time()
        logging.info('processing voice conversion')

        # 添加结果验证
        output_generated = False
        for model_output in self.model.inference(**model_input, stream=stream):
            if 'tts_speech' not in model_output or model_output['tts_speech'] is None:
                logging.warning('Empty or invalid output from model')
                continue

            if model_output['tts_speech'].numel() == 0:
                logging.warning('Empty speech tensor generated')
                continue

            output_generated = True
            speech_len = model_output['tts_speech'].shape[1] / 22050
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()

        if not output_generated:
            raise RuntimeError("No valid output was generated during voice conversion")

    # 字幕生成音频
    def inference_from_srt(self, subtitles, prompt_speech_16k, mode="cross_lingual", language="", prompt_text=None):
        """Generate speech from SRT subtitles using either zero-shot or cross-lingual inference

        Args:
            subtitles (list): List of subtitle objects from SrtPare
            prompt_speech_16k (tensor): Prompt speech tensor (16kHz)
            mode (str): Inference mode - either "zero_shot" or "cross_lingual"
            language (str): Language token (e.g., "<|zh|>", "<|en|>")
            prompt_text (str, optional): Prompt text for zero-shot mode

        Yields:
            dict: Contains generated speech and timing information:
                - tts_speech: Generated speech tensor
                - start_time: Subtitle start time in milliseconds
                - end_time: Subtitle end time in milliseconds
                - text: Original subtitle text
        """
        if mode not in ["zero_shot", "cross_lingual"]:
            raise ValueError("Mode must be either 'zero_shot' or 'cross_lingual'")

        if mode == "zero_shot" and not prompt_text:
            raise ValueError("prompt_text is required for zero_shot mode")

        for subtitle in subtitles:
            text = language + subtitle.content
            start_time = subtitle.start.total_seconds() * 1000
            end_time = subtitle.end.total_seconds() * 1000

            try:
                if mode == "zero_shot":
                    generator = self.inference_zero_shot(text, prompt_text, prompt_speech_16k)
                else:
                    generator = self.inference_cross_lingual(text, prompt_speech_16k)

                for output in generator:
                    output['start_time'] = start_time
                    output['end_time'] = end_time
                    output['text'] = subtitle.content
                    yield output

            except Exception as e:
                logging.error(f"Error processing subtitle: {subtitle.content}")
                logging.error(f"Error details: {str(e)}")
                continue
