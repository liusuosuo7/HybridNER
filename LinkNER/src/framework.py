import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
import time
import prettytable as pt
import os
import json
from metrics.function_metrics import span_f1_prune, ECE_Scores, get_predict_prune
from collections import defaultdict

class FewShotNERFramework:
    def __init__(self, args, logger, task_idx2label, train_data_loader, val_data_loader, test_data_loader, edl, seed_num, num_labels):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger
        self.seed = seed_num
        self.args = args
        self.eps = 1e-10
        self.learning_rate = args.lr
        self.load_ckpt = args.load_ckpt
        self.optimizer = args.optimizer
        self.annealing_start = 1e-6
        self.epoch_num = args.iteration
        self.edl = edl
        self.num_labels = num_labels
        self.task_idx2label = task_idx2label

    def item(self, x):
        return x.item()

    def metric(self, model, eval_dataset, mode):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        context_results = []
        predict_results = []
        prob_results = []
        uncertainty_results = []

        with torch.no_grad():
            for it, data in enumerate(eval_dataset):
                gold_tokens_list = []
                pred_scores_list = []
                pred_list = []
                batch_soft = []

                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = data
                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()
                logits = model(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
                predicts, uncertainty = self.edl.pred(logits)
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(predicts, span_label_ltoken, real_span_mask_ltoken)
                pred_cls, pred_scores, tgt_cls = self.edl.ece_value(logits, span_label_ltoken, real_span_mask_ltoken)

                prob, pred_id = torch.max(predicts, 2)
                batch_results = get_predict_prune(self.args.label2idx_list, all_span_word, words, pred_id, span_label_ltoken, all_span_idxs, prob, uncertainty)
                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct
                batch_soft += span_label_ltoken
                prob_results += prob

                predict_results += pred_id
                uncertainty_results += uncertainty

                context_results += batch_results

                pred_list.append(pred_cls)
                pred_scores_list.append(pred_scores)
                gold_tokens_list.append(tgt_cls)

            gold_tokens_cat = torch.cat(gold_tokens_list, dim=0)
            pred_scores_cat = torch.cat(pred_scores_list, dim=0)
            pred_cat = torch.cat(pred_list, dim=0)

            ece = ECE_Scores(pred_cat, gold_tokens_cat, pred_scores_cat)
            precision = correct_cnt / (pred_cnt + 0.0)
            recall = correct_cnt / (label_cnt + 0.0)
            f1 = 2 * precision * recall / (precision + recall + float("1e-8"))
            if mode == 'test':
                results_dir = os.path.join(self.args.results_dir, f"{self.args.dataname}_{self.args.uncertainty_type}_local_model.jsonl")
                sent_num = len(context_results)

                with open(results_dir, 'w', encoding='utf-8') as fout:
                    for idx in range(sent_num):
                        json.dump(context_results[idx], fout, ensure_ascii=False)
                        fout.write('\n')

                # Optional: Combine with external model outputs
                if getattr(self.args, 'use_combiner', False):
                    comb_files = []
                    if getattr(self.args, 'comb_files', None):
                        comb_files = [x for x in str(self.args.comb_files).split(',') if len(x.strip()) > 0]
                    if len(comb_files) > 0:
                        try:
                            base_path = results_dir
                            combined = self.combine_predictions([base_path] + comb_files,
                                                                method=self.args.comb_method,
                                                                weights=self.args.comb_weights)
                            comb_save = os.path.join(self.args.results_dir, f"{self.args.dataname}_{self.args.uncertainty_type}_combined.jsonl")
                            with open(comb_save, 'w', encoding='utf-8') as fout:
                                for item in combined:
                                    json.dump(item, fout, ensure_ascii=False)
                                    fout.write('\n')
                            self.logger.info(f"Saved combined results to {comb_save}")
                        except Exception as e:
                            self.logger.error(f"Combiner failed: {e}")

            return precision, recall, f1, ece

    def eval(self, model, mode=None):
        if mode == 'dev':
            self.logger.info("Use val dataset")
            precision, recall, f1, ece = self.metric(model, self.val_data_loader, mode='dev')
            self.logger.info('{} Label F1 {}'.format("dev", f1))
            table = pt.PrettyTable(["{}".format("dev"), "Precision", "Recall", 'F1', 'ECE'])

        elif mode == 'test':
            self.logger.info("Use " + str(self.args.test_mode) + " test dataset")
            precision, recall, f1, ece = self.metric(model, self.test_data_loader, mode='test')
            self.logger.info('{} Label F1 {}'.format("test", f1))
            table = pt.PrettyTable(["{}".format("test"), "Precision", "Recall", 'F1', 'ECE'])

        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [precision, recall, f1, ece]])
        self.logger.info("\n{}".format(table))
        return f1, ece

    def train(self, model):
        self.logger.info("Start training...")

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, betas=(0.9, 0.98), lr=self.learning_rate, eps=self.args.adam_epsilon)
        elif self.optimizer == "sgd":
            optimizer = SGD(optimizer_grouped_parameters, self.learning_rate, momentum=0.9)
        elif self.optimizer == "torch.adam":
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon, weight_decay=self.args.weight_decay)

        t_total = len(self.train_data_loader) * self.args.iteration
        warmup_steps = int(self.args.warmup_proportion * t_total)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        model.train()
        best_f1 = 0.0
        best_step = 0
        iter_loss = 0.0

        for idx in range(self.args.iteration):
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0
            epoch_start = time.time()
            self.logger.info("training...")

            for it in range(len(self.train_data_loader)):
                loss = 0
                tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = next(iter(self.train_data_loader))
                loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
                attention_mask = (tokens != 0).long()
                logits = model(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
                loss, pred = self.edl.loss(logits, loadall, span_label_ltoken, real_span_mask_ltoken, idx)
                correct, tmp_pred_cnt, tmp_label_cnt = span_f1_prune(pred, span_label_ltoken, real_span_mask_ltoken)

                pred_cnt += tmp_pred_cnt
                label_cnt += tmp_label_cnt
                correct_cnt += correct

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                iter_loss += self.item(loss.data)

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            precision = correct_cnt / (pred_cnt + 0.)
            recall = correct_cnt / (label_cnt + 0.)
            f1 = 2 * precision * recall / (precision + recall + float("1e-8"))

            self.logger.info("Time '%.2f's" % epoch_cost)
            self.logger.info('step: {0:4} | loss: {1:2.6f} | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'.format(idx + 1, iter_loss, precision, recall, f1))

            if (idx + 1) % 1 == 0:
                f1, ece = self.eval(model, mode='dev')
                self.inference(model)
                if f1 > best_f1:
                    best_step = idx + 1
                    best_f1 = f1
                    if self.args.load_ckpt:
                        torch.save(model, self.args.results_dir + self.args.loss + str(self.args.seed) + '_model.pkl')

                if (idx + 1) > best_step + self.args.early_stop:
                    self.logger.info('Early stop!')
                    return

            iter_loss = 0.
            pred_cnt = 0
            label_cnt = 0
            correct_cnt = 0

    def inference(self, model):
        f1, ece = self.eval(model, mode='test')

    def combine_predictions(self, files, method='majority', weights=None):
        # Load all files as list of dicts per line
        lists = []
        for fp in files:
            with open(fp, 'r', encoding='utf-8') as f:
                lists.append([json.loads(line) for line in f])
        # Ensure same length and sentence alignment
        base = lists[0]
        num = len(base)
        for lst in lists[1:]:
            if len(lst) != num:
                raise ValueError('Combiner: input files size mismatch')
        # Parse weights
        w = None
        if method == 'weighted' and weights:
            w_vals = [float(x) for x in str(weights).split(',')]
            if len(w_vals) == len(files):
                w = w_vals
            elif len(w_vals) == len(files) - 1:
                w = [1.0] + w_vals
            else:
                raise ValueError('Combiner: weights length must match number of files or number of external files')
        # Label vote per span key (sidx,eidx)
        combined = []
        for i in range(num):
            sent = base[i]['sentence']
            # Build union of spans across models
            span_keys = set()
            for lst in lists:
                for ent in lst[i].get('entity', []):
                    if 'span' in ent and isinstance(ent['span'], list) and len(ent['span']) == 2:
                        span_keys.add(tuple(ent['span']))
            entities = []
            for span in sorted(span_keys):
                label_scores = defaultdict(float)
                best_conf = 0.0
                best_unc = 1.0
                ent_text = None
                for m_idx, lst in enumerate(lists):
                    # find matching ent in this model output
                    found = None
                    for ent in lst[i].get('entity', []):
                        if tuple(ent.get('span', [])) == span:
                            found = ent
                            break
                    lab = found.get('llm_pred') if found and found.get('llm_pred') else (found.get('pred') if found else 'O')
                    if isinstance(lab, list) and len(lab) > 0:
                        lab = lab[0]
                    weight = 1.0
                    if w is not None:
                        weight = w[m_idx] if m_idx < len(w) else 1.0
                    label_scores[lab] += weight
                    if found:
                        ent_text = found.get('entity', ent_text)
                        best_conf = max(best_conf, float(found.get('confidence', 0.0)))
                        best_unc = min(best_unc, float(found.get('uncertainty', 1.0)))
                # choose label (ignore O unless all are O)
                if len(label_scores) == 0:
                    continue
                # ensure 'O' does not dominate if others exist
                if 'O' in label_scores and len(label_scores) > 1:
                    del label_scores['O']
                final_lab = max(label_scores.items(), key=lambda kv: kv[1])[0]
                if final_lab == 'O':
                    continue
                # find an answer label from any model output for this span
                answer_lab = 'O'
                for lst in lists:
                    for ent in lst[i].get('entity', []):
                        if tuple(ent.get('span', [])) == span and 'answer' in ent:
                            ans = ent.get('answer')
                            if isinstance(ans, list) and len(ans) > 0:
                                ans = ans[0]
                            answer_lab = ans
                            break
                    if answer_lab != 'O':
                        break
                res_val = 1 if final_lab == answer_lab else 0
                entities.append({
                    'entity': ent_text if ent_text is not None else '',
                    'span': [int(span[0]), int(span[1])],
                    'pred': final_lab,
                    'answer': answer_lab,
                    'confidence': best_conf,
                    'uncertainty': best_unc,
                    'res': res_val
                })
            combined.append({'sentence': sent, 'entity': entities})
        return combined