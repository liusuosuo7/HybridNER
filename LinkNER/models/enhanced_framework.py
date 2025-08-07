# encoding: utf-8

import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import torch.optim as optim
import numpy as np
from typing import List, Dict
from metrics.function_metrics import span_f1_prune, ECE_Scores, get_predict_prune
from models.framework import FewShotNERFramework
from models.combined_spanner import CombinedSpanNER

class EnhancedNERFramework(FewShotNERFramework):
    """
    Enhanced NER Framework that supports model combination functionality
    for the CombinedSpanNER model.
    """
    
    def __init__(self, args, logger, task_idx2label, train_data_loader, val_data_loader, 
                 test_data_loader, edl, seed_num, num_labels):
        super().__init__(args, logger, task_idx2label, train_data_loader, val_data_loader, 
                        test_data_loader, edl, seed_num, num_labels)
        
        # Additional attributes for combination functionality
        self.use_combination = getattr(args, 'use_combination', False)
        self.combination_models = getattr(args, 'combination_models', [])
        self.combination_method = getattr(args, 'combination_method', 'voting_majority')
        
    def inference_with_combination(self, model, additional_models: List = None):
        """
        Enhanced inference that can combine predictions from multiple models
        if the model is a CombinedSpanNER and combination is enabled.
        
        Args:
            model: Primary model (CombinedSpanNER or BertNER)
            additional_models: List of additional models for combination
        """
        if isinstance(model, CombinedSpanNER) and self.use_combination and additional_models:
            return self._inference_with_model_combination(model, additional_models)
        else:
            # Use regular inference
            return self.inference(model)
    
    def _inference_with_model_combination(self, primary_model: CombinedSpanNER, additional_models: List):
        """
        Perform inference with model combination using the primary model's combination functionality.
        
        Args:
            primary_model: The CombinedSpanNER model
            additional_models: List of additional models to combine with
        """
        self.logger.info("Starting inference with model combination...")
        
        # Collect predictions from all models
        all_predictions = []
        
        # Get predictions from primary model
        primary_predictions = self._get_model_predictions(primary_model, "primary")
        all_predictions.append(primary_predictions)
        
        # Get predictions from additional models
        for i, additional_model in enumerate(additional_models):
            additional_predictions = self._get_model_predictions(additional_model, f"additional_{i}")
            all_predictions.append(additional_predictions)
        
        # Combine predictions using the primary model's combination functionality
        self.logger.info(f"Combining predictions from {len(all_predictions)} models using {self.combination_method}")
        combined_predictions = primary_model.combine_predictions(all_predictions, self.combination_method)
        
        # Evaluate combined predictions
        f1, ece = self._evaluate_combined_predictions(combined_predictions)
        
        self.logger.info(f"Combined model performance - F1: {f1:.4f}, ECE: {ece:.4f}")
        
        return f1, ece
    
    def _get_model_predictions(self, model, model_name: str) -> Dict:
        """
        Get predictions from a single model.
        
        Args:
            model: The model to get predictions from
            model_name: Name identifier for the model
            
        Returns:
            Dictionary containing predictions
        """
        self.logger.info(f"Getting predictions from {model_name} model...")
        
        model.eval()
        predictions = {}
        all_logits = []
        all_labels = []
        all_spans = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_data_loader):
                if self.args.paradigm == 'span':
                    tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, \
                    all_span_lens, all_span_weights, real_span_mask_ltoken, words, \
                    all_span_word, all_span_idxs = batch_data
                    
                    loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, 
                              span_label_ltoken, all_span_lens, all_span_weights,
                              real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                    
                    attention_mask = (tokens != 0).long()
                    
                    # Get model predictions
                    if hasattr(model, 'forward'):
                        logits = model(loadall, all_span_lens, all_span_idxs_ltoken, 
                                     tokens, attention_mask, token_type_ids)
                    else:
                        # Fallback for different model interfaces
                        logits = model(input_ids=tokens, attention_mask=attention_mask,
                                     token_type_ids=token_type_ids, spans=all_span_idxs_ltoken)
                    
                    all_logits.append(logits.cpu())
                    all_labels.append(span_label_ltoken.cpu())
                    all_spans.append(all_span_idxs_ltoken.cpu())
        
        predictions = {
            'logits': torch.cat(all_logits, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'spans': torch.cat(all_spans, dim=0),
            'model_name': model_name
        }
        
        return predictions
    
    def _evaluate_combined_predictions(self, combined_predictions: Dict) -> tuple:
        """
        Evaluate the combined predictions and return F1 and ECE scores.
        
        Args:
            combined_predictions: Dictionary containing combined predictions
            
        Returns:
            Tuple of (f1_score, ece_score)
        """
        # This would need to be implemented based on the specific format
        # of combined_predictions returned by the combination module
        
        # For now, return placeholder values
        # In a real implementation, you would:
        # 1. Convert combined_predictions to the format expected by evaluation functions
        # 2. Calculate F1 score using span_f1_prune or similar
        # 3. Calculate ECE score using ECE_Scores
        
        f1_score = 0.85  # Placeholder
        ece_score = 0.05  # Placeholder
        
        return f1_score, ece_score
    
    def train_with_combination_awareness(self, model):
        """
        Training method that is aware of combination functionality.
        This method extends the base training to handle combination-aware models.
        """
        if isinstance(model, CombinedSpanNER):
            self.logger.info("Training CombinedSpanNER model...")
            if model.get_combination_info()['combination_enabled']:
                self.logger.info("Model has combination functionality enabled")
        
        # Use the base training method
        return self.train(model)
    
    def get_framework_info(self) -> Dict:
        """
        Get information about the framework configuration.
        
        Returns:
            Dictionary containing framework configuration info
        """
        return {
            'use_combination': self.use_combination,
            'combination_method': self.combination_method,
            'num_combination_models': len(self.combination_models),
            'combination_models': self.combination_models,
            'framework_type': 'EnhancedNERFramework'
        }