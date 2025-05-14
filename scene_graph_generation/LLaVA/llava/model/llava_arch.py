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


from abc import ABC, abstractmethod
from typing import Optional, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, VIS_DESCRIPTOR_TOKEN_INDEX

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_image_pooler


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            self.image_pooler = build_image_pooler(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_image_pooler(self):
        return self.image_pooler

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # unfreeze image pooler
        for p in self.image_pooler.parameters():
            p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_image_pooler(self):
        return self.get_model().get_image_pooler()
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def visualize_feature_maps(self, image_features, num_images=1, num_features=1):
        # Convert to float32 for visualization
        image_features = image_features.to(dtype=torch.float32)

        fig, axes = plt.subplots(1, num_images, figsize=(15, 6))

        if num_images == 1:  # This is to handle the case when there's only one image
            axes = [axes]

        for i in range(num_images):
            for j in range(num_features):
                # Visualize the j-th feature of the i-th image
                ax = axes[i]
                ax.imshow(image_features[i, j, :, :].detach().cpu(), cmap='viridis')
                ax.set_title(f"Image {i + 1}, Feature {j + 1}")
                ax.axis('off')

        plt.tight_layout()
        plt.show()

    def encode_images_concat_pool(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        # switch last two dimensions
        image_features = image_features.permute(0, 2, 1)
        # reshape to num_imgs x hidden_size x 24 x 24
        image_features = image_features.reshape((image_features.shape[0], image_features.shape[1], 24, 24))
        # max pooling
        image_features = F.max_pool2d(image_features, kernel_size=2, stride=2)
        # flatten to num_imgs x hidden_size x num_patches
        image_features = image_features.reshape((image_features.shape[0], image_features.shape[1], -1))
        # switch last two dimensions back
        image_features = image_features.permute(0, 2, 1)
        return image_features
    
    
    def pad_embeddings(self, embeddings, padding_value=0):
        """
        Pad the embeddings to have the same number in each batch.

        Args:
        - embeddings (List[Tensor]): List of embedding tensors, each with shape (num_images, embedding_dim).
        - padding_value (float): Value to use for padding.

        Returns:
        - Tensor: Padded embeddings with shape (batch_size, max_num_images, embedding_dim).
        - Tensor: Mask indicating real data (1) and padding (0).
        """
        batch_size = len(embeddings)
        img_len = embeddings[0].shape[1]
        embedding_dim = embeddings[0].shape[2]
        max_num_images = max(emb.shape[0] for emb in embeddings)

        # Initialize padded embeddings and mask
        padded_embeddings = torch.full((batch_size, max_num_images, img_len, embedding_dim), padding_value, dtype=embeddings[0].dtype, device=embeddings[0].device)
        mask = torch.zeros(batch_size, max_num_images * img_len, dtype=torch.bool, device=embeddings[0].device)

        # Pad each item in the batch
        for idx, emb in enumerate(embeddings):
            num_images = emb.shape[0]
            padded_embeddings[idx, :num_images] = emb
            mask[idx, :num_images * img_len] = 1

        return padded_embeddings.flatten(1, 2), mask

    def encode_images_pooled(self, ego_images=None, exo_images=None, split_sizes=None, 
                             gaze=None, gaze_depth=None, 
                             hand_tracking=None, hand_mask=None, 
                             audio=None, 
                             point_cloud=None, 
                             ego_camera_ids=None, exo_camera_ids=None
        ):

        image_pooler = self.get_image_pooler()
        ego_image_features = self.get_model().get_vision_tower()(ego_images) if ego_images is not None else None
        exo_image_features = self.get_model().get_vision_tower()(exo_images) if exo_images is not None else None
        ego_split_sizes = split_sizes[0]
        exo_split_sizes = split_sizes[1]

        if ego_image_features is not None:
            if ego_split_sizes is not None:
                ego_image_features = torch.split(ego_image_features, ego_split_sizes, dim=0)
                ego_image_features, ego_image_mask = self.pad_embeddings(ego_image_features)
                ego_image_features = image_pooler(
                    embeddings=ego_image_features,
                    mask=ego_image_mask,
                    gaze=gaze,
                    gaze_depth=gaze_depth,
                    hand_tracking=hand_tracking,
                    split_sizes=ego_split_sizes,
                )
            else:
                ego_image_mask = torch.ones((ego_image_features.shape[0], ego_image_features.shape[1]), dtype=torch.bool, device=ego_image_features.device)
                ego_image_features = image_pooler(
                    embeddings=ego_image_features,
                    mask=ego_image_mask,
                    gaze=gaze,
                    gaze_depth=gaze_depth,
                    hand_tracking=hand_tracking,
                    split_sizes=ego_split_sizes
                )
        if exo_image_features is not None:
            if exo_split_sizes is not None:
                exo_image_features = torch.split(exo_image_features, exo_split_sizes, dim=0)
                exo_image_features, exo_image_mask = self.pad_embeddings(exo_image_features)
                exo_image_features = image_pooler(
                    embeddings=exo_image_features,
                    mask=exo_image_mask,
                    audio=audio,
                    point_cloud=point_cloud,
                    split_sizes=exo_split_sizes
                )
            else:
                exo_image_mask = torch.ones((exo_image_features.shape[0], exo_image_features.shape[1]), dtype=torch.bool, device=exo_image_features.device)
                exo_image_features = image_pooler(
                    embeddings=exo_image_features,
                    mask=exo_image_mask,
                    audio=audio,
                    point_cloud=point_cloud,
                    split_sizes=exo_split_sizes
                )

        if ego_image_features is None and exo_image_features is not None:
            image_features = exo_image_features
        elif ego_image_features is not None and exo_image_features is None:
            image_features = ego_image_features
        elif ego_image_features is None and exo_image_features is None:
            raise ValueError("Both ego and exo image features are None.")
        else:
            image_features = torch.cat([ego_image_features, exo_image_features], dim=1)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    

    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, 
        ego_frames, exo_frames, eye_gaze, eye_gaze_depth, hand_tracking, audio, point_cloud,
        ego_source_ids=None, exo_source_ids=None, vis_descriptor_embs=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or (ego_frames is None and exo_frames is None) or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and (ego_frames is not None or exo_frames is not None) and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        # Prepare images for encoding
        has_list_ego = isinstance(ego_frames, list)
        has_list_exo = isinstance(exo_frames, list)
        has_nd_ego   = hasattr(ego_frames, "ndim") and ego_frames.ndim == 5
        has_nd_exo   = hasattr(exo_frames, "ndim") and exo_frames.ndim == 5
        if has_list_ego or has_nd_ego or has_list_exo or has_nd_exo:
            if getattr(self.config, 'mv_type') == "learned":
                concat_ego_images = torch.cat([ego_frame for ego_frame in ego_frames], dim=0) if ego_frames is not None else None
                split_ego_sizes = [ego_frame.shape[0] for ego_frame in ego_frames] if ego_frames is not None else None

                concat_exo_images = torch.cat([exo_frame for exo_frame in exo_frames], dim=0) if exo_frames is not None else None
                split_exo_sizes = [exo_frame.shape[0] for exo_frame in exo_frames] if exo_frames is not None else None

                concat_gaze = torch.cat([eye_gaze["data"] for eye_gaze in eye_gaze], dim=0) if eye_gaze is not None else None # -->> B*N_camera, 2 : e.g. 4*4 , 2 = [4,16]
                concat_gaze_depth = torch.cat([eye_gaze_depth["data"].unsqueeze(-1) for eye_gaze_depth in eye_gaze_depth], dim=0) if eye_gaze_depth is not None else None # -->> B*N_camera, 1 : e.g. [4*4] = [16]
                concat_hand_tracking = torch.cat([hand_tracking["data"] for hand_tracking in hand_tracking], dim=0) if hand_tracking is not None else None # -->> B*N_camera, 16 : e.g. 4*4, 16
                concat_hand_mask = torch.cat([hand_tracking["mask"] for hand_tracking in hand_tracking], dim=0) if hand_tracking is not None else None # -->> B*N_camera, 1 : e.g. 4*4 : [16]
                concat_audio = torch.cat([audio_embed["data"].unsqueeze(0) for audio_embed in audio], dim=0) if audio is not None else None # --> B, audio_dim, stereo_channel : e.g. [4, 4800, 2], since we already saved merged audio snippets, each sample in a batch will have only one global audio snippet.
                
                if point_cloud is not None:
                    concat_pc = [
                        (pc["data"].to(attention_mask.device)
                         if pc is not None else None)
                        for pc in point_cloud
                    ]
                else:
                    concat_pc = None

                split_sizes = (split_ego_sizes, split_exo_sizes)

                image_features = self.encode_images_pooled(
                    concat_ego_images, concat_exo_images, split_sizes,
                    concat_gaze, concat_gaze_depth, concat_hand_tracking, concat_hand_mask, concat_audio, concat_pc,
                    ego_source_ids, exo_source_ids
                )
        else:
            raise Exception('SHOULD NOT BE HERE')

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_vis_descriptors = (cur_input_ids == VIS_DESCRIPTOR_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where((cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == VIS_DESCRIPTOR_TOKEN_INDEX))[0].tolist() + [
                cur_input_ids.shape[0]]  # first one is image, rest are vis descriptors
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            # add visual descriptors in order
            # image loop already added first text part after the <image> token, so we can directly add the first vis descriptor + the text part after it
            if vis_descriptor_embs is not None:
                if type(vis_descriptor_embs[0]) is not list:  # batchsize 1
                    vis_descriptor_embs = [vis_descriptor_embs]
                for j in range(num_vis_descriptors):
                    if j < len(vis_descriptor_embs[batch_idx]):
                        cur_descriptor_features = vis_descriptor_embs[batch_idx][j].to(self.device)
                    else:
                        print(f"Batch idx: {batch_idx}, j: {j}, num_vis_descriptors: {num_vis_descriptors} and len(vis_descriptor_embs): {len(vis_descriptor_embs)}")
                        print('Using dummy tensor')
                        cur_descriptor_features = torch.zeros(4096, device=image_features.device, dtype=image_features.dtype)

                    if len(cur_descriptor_features.shape) == 1:
                        cur_descriptor_features = cur_descriptor_features.unsqueeze(0)
                    cur_new_input_embeds.append(cur_descriptor_features)
                    cur_new_labels.append(torch.full((cur_descriptor_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_input_embeds.append(cur_input_embeds_no_im[num_images + j + 1])
                    cur_new_labels.append(cur_labels_noim[num_images + j + 1])

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
