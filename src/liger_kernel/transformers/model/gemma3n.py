from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.cache_utils import HybridCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss


def causal_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[HybridCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    skip_logits: Optional[bool] = None,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    Gemma3n text-only forward with optional fused linear + cross entropy.

    - Matches Gemma3n's logits path, including the final tanh soft-cap.
    - When `skip_logits` is True (or training with labels), computes loss via
      Liger fused linear+CE without materializing logits.
    - Otherwise, computes logits normally and (optionally) loss via CE.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Decoder outputs: (last_hidden_state, past_key_values, hidden_states, attentions)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **loss_kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    kept_hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None

    # Default to fused path in training when labels are provided unless overridden
    if skip_logits is None:
        shift_labels = loss_kwargs.get("shift_labels", None)
        skip_logits = self.training and (labels is not None or shift_labels is not None)

    # Final logit soft-cap (text-only config stores this directly on self.config)
    final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)

    if skip_logits:
        # Fused linear + CE, no logits materialization
        loss = LigerForCausalLMLoss(
            hidden_states=kept_hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            shift_labels=loss_kwargs.pop("shift_labels", None),
            hidden_size=self.config.hidden_size,
            final_logit_softcapping=final_logit_softcapping,
            **loss_kwargs,
        )
    else:
        # Standard logits path + optional loss using HF's loss_function to match semantics
        logits = self.lm_head(kept_hidden_states)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        if labels is not None:
            # Defer to the model's built-in loss function to preserve HF behavior
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
