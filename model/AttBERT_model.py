import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from model.tempobert.modeling_tempobert import TempoBertForMaskedLM

def load_AttBERT_model(logger, max_length, vocab_size, pad_token_id, 
                       model_args, times, time_embedding_type="temporal_attention"):
    logger.info("Loading vanilla BERT model")
    config = BertConfig(
        **model_args,
        max_position_embeddings=max_length,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id)
    bert_model = BertForMaskedLM(config)

    logger.info("Loading TempoBERT model")
    config.times = times
    config.time_embedding_type = time_embedding_type
    model = TempoBertForMaskedLM.from_non_temporal(bert_model, config)
    
    return model

class DMSAttBERT(nn.Module):
    def __init__(self, AttBERT, emb_dim):
        super(DMSAttBERT, self).__init__()
        self.AttBERT = AttBERT
        self.predict_module = nn.Sequential(
            nn.Linear(self.AttBERT.config.hidden_size, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 1)
        )
    
    def forward(self, sequence, site_list):
        sequence_encoded = self.AttBERT(**sequence).last_hidden_state
        # sequence_cls = sequence_encoded[list(range(sequence_encoded.shape[0])), site_list, :]
        # sum along the second dimension
        sequence_cls = sequence_encoded.sum(dim=1)
        prediction = self.predict_module(sequence_cls)
        return prediction
    

class AttBERTForRegression(PreTrainedModel):
    def __init__(self, AttBERT, n_targets, intermediate_dim=256, dropout_rate=0.1):
        config = BertConfig()
        super(AttBERTForRegression, self).__init__(config)
        self.AttBERT = AttBERT
        self.predict_module = nn.Sequential(
            nn.Linear(self.AttBERT.config.hidden_size, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, n_targets))
    
    # weighted_mse_loss
    def weighted_mse_loss(self, inputs, targets, weights):
        diff = inputs - targets
        diff_squared = diff ** 2
        weighted_diff_squared = diff_squared * weights
        loss = weighted_diff_squared.mean()
        return loss

    def forward(self, input_ids, attention_mask=None, labels=None, weights=None, **kwargs):
        outputs = self.AttBERT(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0]
        regression_output = self.predict_module(pooled_output)

        if weights is None:
            weights = torch.ones_like(regression_output)

        #calculating loss only for tasks included in a data chunk
        loss = None
        if labels is not None:
            mask = ~torch.isnan(labels)
            masked_labels = labels[mask]
            masked_regression_output = regression_output[mask]
            masked_weights = weights[mask]
            loss = self.weighted_mse_loss(masked_regression_output, masked_labels, masked_weights)

            return SequenceClassifierOutput(
                loss=loss,
                logits=regression_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions)

        else:
            return SequenceClassifierOutput(
                logits=regression_output,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions)