#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        ego_frames: Optional[torch.FloatTensor] = None,
        exo_frames: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        eye_gaze=None,
        eye_gaze_depth=None,
        hand_tracking=None,
        audio=None,
        point_cloud=None,
        ego_source_ids=None,
        exo_source_ids=None,
        ego_source_names=None,
        exo_source_names=None,
        vis_descriptor_embs=None

    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                ego_frames, # ego_images
                exo_frames, # exo_images
                eye_gaze, # gaze
                eye_gaze_depth, # gaze_depth
                hand_tracking, # hand_tracking
                audio, # audio
                point_cloud, # point_cloud
                ego_source_ids,
                exo_source_ids,
                vis_descriptor_embs

            )
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        output['modified_labels'] = labels
        return output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        ego_frames = kwargs.pop("ego_frames", None)
        exo_frames = kwargs.pop("exo_frames", None)
        eye_gaze = kwargs.pop("eye_gaze", None)
        eye_gaze_depth = kwargs.pop("eye_gaze_depth", None)
        hand_tracking = kwargs.pop("hand_tracking", None)
        audio = kwargs.pop("audio", None)
        point_cloud = kwargs.pop("point_cloud", None)
        ego_source_ids = kwargs.pop("ego_source_ids", None)
        exo_source_ids = kwargs.pop("exo_source_ids", None)

        # Call the parent class's method to prepare the base inputs
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        # Add multimodal inputs to the dictionary if they are provided
        if ego_frames is not None:
            _inputs['ego_frames'] = ego_frames
        if exo_frames is not None:
            _inputs['exo_frames'] = exo_frames
        if eye_gaze is not None:
            _inputs['eye_gaze'] = eye_gaze
        if eye_gaze_depth is not None:
            _inputs['eye_gaze_depth'] = eye_gaze_depth
        if hand_tracking is not None:
            _inputs['hand_tracking'] = hand_tracking
        if audio is not None:
            _inputs['audio'] = audio
        if point_cloud is not None:
            _inputs['point_cloud'] = point_cloud
        if ego_source_ids is not None:
            _inputs['ego_source_ids'] = ego_source_ids
        if exo_source_ids is not None:
            _inputs['exo_source_ids'] = exo_source_ids

        return _inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
