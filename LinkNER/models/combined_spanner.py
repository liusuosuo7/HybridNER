# encoding: utf-8

import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple
from transformers import BertModel, BertPreTrainedModel, RobertaModel
from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F
import sys
import json

# Import combination functionality from spanNER
sys.path.append(os.path.join(os.path.dirname(__file__), '../../spanNER/combination'))
from comb_voting import CombByVoting
from dataread import DataReader

class CombinedSpanNER(BertPreTrainedModel):
    """
    Combined SpanNER model that integrates multiple spanNER models and other sequence models
    through voting strategies for enhanced NER performance.
    """
    
    def __init__(self, config, args):
        super(CombinedSpanNER, self).__init__(config)
        
        # Initialize base spanNER model
        self.bert = BertModel(config)
        self.args = args
        
        if 'roberta' in self.args.bert_config_dir:
            self.bert = RobertaModel(config)
            print('Using RoBERTa pre-trained model...')
            
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        self.end_outputs = nn.Linear(config.hidden_size, 1)
        
        if torch.cuda.is_available():
            self.start_outputs = self.start_outputs.cuda()
            self.end_outputs = self.end_outputs.cuda()
        
        self.hidden_size = config.hidden_size
        self.span_combination_mode = self.args.span_combination_mode
        self.max_span_width = args.max_spanLen
        self.n_class = args.n_class
        self.tokenLen_emb_dim = self.args.tokenLen_emb_dim
        
        print("Max span width: ", self.max_span_width)
        print("Token length embedding dimension: ", self.tokenLen_emb_dim)
        
        self._endpoint_span_extractor = EndpointSpanExtractor(
            config.hidden_size,
            combination=self.span_combination_mode,
            num_width_embeddings=self.max_span_width,
            span_width_embedding_dim=self.tokenLen_emb_dim,
            bucket_widths=True
        )
        
        # Combination-specific attributes
        self.combination_models = getattr(args, 'combination_models', [])
        self.combination_method = getattr(args, 'combination_method', 'voting_majority')
        self.combination_classes = getattr(args, 'combination_classes', [])
        self.combination_results_dir = getattr(args, 'combination_results_dir', '')
        self.combination_prob_file = getattr(args, 'combination_prob_file', '')
        self.combination_standard_file = getattr(args, 'combination_standard_file', '')
        
        # Initialize combination module if models are specified
        if self.combination_models:
            self.init_combination_module()
        
        # Classification layers
        self.linear = nn.Linear(10, 1)
        self.score_func = nn.Softmax(dim=-1)
        
        feature_dim = self._endpoint_span_extractor.get_output_dim()
        
        if self.args.use_spanLen:
            feature_dim += self.args.spanLen_emb_dim
            self.spanLen_emb = nn.Embedding(self.max_span_width + 1, self.args.spanLen_emb_dim)
            
        if self.args.use_morph:
            feature_dim += self.args.morph_emb_dim
            morph_vocab_size = len(self.args.morph2idx_list)
            self.morph_emb = nn.Embedding(morph_vocab_size, self.args.morph_emb_dim)
            
        if self.args.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(feature_dim, self.n_class)
        else:
            self.classifier = MultiNonLinearClassifier(
                feature_dim, self.n_class, self.args.model_dropout,
                self.args.classifier_act_func, intermediate_hidden_size=feature_dim
            )
            
        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()
            
    def init_combination_module(self):
        """Initialize the combination module with specified models."""
        print(f"Initializing combination with {len(self.combination_models)} models")
        print(f"Combination method: {self.combination_method}")
        
        # Extract F1 scores from model filenames (assuming they follow naming convention)
        self.combination_f1s = []
        for model_file in self.combination_models:
            try:
                # Extract F1 from filename pattern like "model_9241.txt"
                f1_str = model_file.split('_')[-1].split('.')[0]
                f1 = float(f1_str) / 10000.0
                self.combination_f1s.append(f1)
            except:
                # Default F1 if pattern doesn't match
                self.combination_f1s.append(0.85)
                
        print(f"Model F1 scores: {self.combination_f1s}")
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
                spans=None, spans_mask=None, span_labels=None, **kwargs):
        """Forward pass through the combined model."""
        
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        
        # Extract span representations
        span_embeddings = self._endpoint_span_extractor(sequence_output, spans)
        
        # Add additional features if enabled
        if self.args.use_spanLen:
            spans_length = spans[:, :, 1] - spans[:, :, 0] + 1
            spans_length = torch.clamp(spans_length, min=0, max=self.max_span_width)
            span_length_embeddings = self.spanLen_emb(spans_length)
            span_embeddings = torch.cat([span_embeddings, span_length_embeddings], dim=-1)
            
        if self.args.use_morph:
            spans_morph = kwargs.get('spans_morph', None)
            if spans_morph is not None:
                span_morph_embeddings = self.morph_emb(spans_morph)
                span_embeddings = torch.cat([span_embeddings, span_morph_embeddings], dim=-1)
                
        # Classification
        span_logits = self.classifier(span_embeddings)
        
        return span_logits
    
    def combine_predictions(self, predictions_list: List[Dict], method: str = None) -> Dict:
        """
        Combine predictions from multiple models using specified voting strategy.
        
        Args:
            predictions_list: List of prediction dictionaries from different models
            method: Combination method ('voting_majority', 'voting_weightByOverallF1', 
                   'voting_weightByCategotyF1', 'voting_spanPred_onlyScore')
        
        Returns:
            Combined predictions dictionary
        """
        if not self.combination_models or len(predictions_list) <= 1:
            return predictions_list[0] if predictions_list else {}
            
        method = method or self.combination_method
        
        # Save predictions to temporary files for combination module
        temp_dir = "/tmp/linkner_combination"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create temporary result files
        temp_files = []
        for i, pred_dict in enumerate(predictions_list):
            temp_file = os.path.join(temp_dir, f"temp_model_{i}.txt")
            self._save_predictions_to_file(pred_dict, temp_file)
            temp_files.append(f"temp_model_{i}.txt")
            
        # Use combination module
        try:
            comvote = CombByVoting(
                dataname=self.args.dataname,
                file_dir=temp_dir,
                fmodels=temp_files,
                f1s=self.combination_f1s[:len(predictions_list)],
                cmodelname="combined_linkner",
                classes=self.combination_classes,
                fn_stand_res=temp_files[0],  # Use first model as standard
                fn_prob=self.combination_prob_file
            )
            
            # Apply selected combination method
            if method == 'voting_majority':
                result = comvote.voting_majority()
            elif method == 'voting_weightByOverallF1':
                result = comvote.voting_weightByOverallF1()
            elif method == 'voting_weightByCategotyF1':
                result = comvote.voting_weightByCategotyF1()
            elif method == 'voting_spanPred_onlyScore':
                result = comvote.voting_spanPred_onlyScore()
            else:
                print(f"Unknown combination method: {method}, using majority voting")
                result = comvote.voting_majority()
                
            # Convert result back to prediction format
            combined_predictions = self._convert_result_to_predictions(result, comvote)
            
        except Exception as e:
            print(f"Error in combination: {e}")
            print("Falling back to first model predictions")
            combined_predictions = predictions_list[0]
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                full_path = os.path.join(temp_dir, temp_file)
                if os.path.exists(full_path):
                    os.remove(full_path)
                    
        return combined_predictions
    
    def _save_predictions_to_file(self, predictions: Dict, filepath: str):
        """Save predictions in the format expected by combination module."""
        # This would need to be implemented based on the specific format
        # expected by the combination module
        pass
    
    def _convert_result_to_predictions(self, combination_result: List, comvote: CombByVoting) -> Dict:
        """Convert combination result back to prediction dictionary format."""
        # This would need to be implemented based on the specific format
        # of your prediction dictionaries
        pass
    
    def get_combination_info(self) -> Dict:
        """Get information about the combination setup."""
        return {
            'combination_enabled': bool(self.combination_models),
            'num_models': len(self.combination_models),
            'combination_method': self.combination_method,
            'model_f1s': self.combination_f1s if hasattr(self, 'combination_f1s') else [],
            'classes': self.combination_classes
        }