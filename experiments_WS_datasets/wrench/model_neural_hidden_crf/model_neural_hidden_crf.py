import logging
import math
from copy import deepcopy
from typing import Any, List, Optional, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import trange
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from ..backbone import BertSeqTagger
from ..basemodel import BaseTorchSeqModel
from ..dataset.seqdataset import BaseSeqDataset
from ..utils import construct_collate_fn_trunc_pad
import os

logger = logging.getLogger(__name__)
collate_fn = construct_collate_fn_trunc_pad('mask')




class BERTTorchSeqDataset(Dataset):
    def __init__(self, dataset: BaseSeqDataset, tokenizer, max_seq_length, use_crf, n_data: Optional[int] = 0):
        self.id2label = deepcopy(dataset.id2label)
        self.label2id = deepcopy(dataset.label2id)
        self.n_class = len(self.id2label)

        if not use_crf:
            self.dum_label = 'X'
            self.label2id[self.dum_label] = len(self.id2label)
            self.id2label.append(self.dum_label)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length  # set to -1 when test
        self.use_crf = use_crf

        corpus = list(map(lambda x: x["text"], dataset.examples))
        self.seq_len = list(map(len, corpus))
        input_ids_tensor, input_mask_tensor, predict_mask_tensor = self.convert_corpus_to_tensor(corpus)

        self.input_ids_tensor = input_ids_tensor
        self.input_mask_tensor = input_mask_tensor
        self.predict_mask_tensor = predict_mask_tensor

        n_data_ = len(input_ids_tensor)
        self.n_data_ = n_data_
        if n_data > 0:
            self.n_data = math.ceil(n_data / n_data_) * n_data_
        else:
            self.n_data = n_data_

    def __len__(self):
        return self.n_data

    def convert_corpus_to_tensor(self, corpus):
        input_ids_list = []
        input_mask_list = []
        predict_mask_list = []
        max_seq_length = 0

        for words in corpus:
            predict_mask = []
            input_mask = []
            tokens = []

            for i, w in enumerate(words):
                sub_words = self.tokenizer.tokenize(w)
                if not sub_words:
                    sub_words = [self.tokenizer.unk_token]
                if self.use_crf:
                    ''' if crf is used, then the padded token will be ignored '''
                    tokens.append(sub_words[0])
                else:
                    tokens.extend(sub_words)
                for j in range(len(sub_words)):
                    if j == 0:
                        input_mask.append(1)
                        predict_mask.append(1)
                    elif not self.use_crf:  # These padding will hurt performance
                        ''' '##xxx' -> 'X' (see bert paper, for non-crf model only) '''
                        input_mask.append(1)
                        predict_mask.append(0)

            max_seq_length = max(max_seq_length, len(tokens))
            input_ids_list.append(self.tokenizer.convert_tokens_to_ids(tokens))
            input_mask_list.append(input_mask)
            predict_mask_list.append(predict_mask)

        max_seq_length = min(max_seq_length, self.max_seq_length)

        n = len(input_ids_list)
        for i in range(n):
            ni = len(input_ids_list[i])
            if ni > max_seq_length:
                logger.info(f'Example is too long, length is {ni}, truncated to {max_seq_length}!')
                input_ids_list[i] = input_ids_list[i][:max_seq_length]
                input_mask_list[i] = input_mask_list[i][:max_seq_length]
                predict_mask_list[i] = predict_mask_list[i][:max_seq_length]
            else:
                input_ids_list[i].extend([self.tokenizer.pad_token_id] * (max_seq_length - ni))
                input_mask_list[i].extend([0] * (max_seq_length - ni))
                predict_mask_list[i].extend([0] * (max_seq_length - ni))

        input_ids_tensor = torch.LongTensor(input_ids_list)
        input_mask_tensor = torch.LongTensor(input_mask_list)
        predict_mask_tensor = torch.LongTensor(predict_mask_list)

        return input_ids_tensor, input_mask_tensor, predict_mask_tensor

    def prepare_labels(self, labels):
        O_id = self.label2id['O']
        if self.use_crf:
            n, max_seq_len = self.predict_mask_tensor.shape
            prepared_labels = np.ones((n, max_seq_len), dtype=int) * O_id
            for i, labels_i in enumerate(labels):
                ni = len(labels_i)
                if ni > max_seq_len:
                    prepared_labels[i, :] = labels_i[:max_seq_len]
                else:
                    prepared_labels[i, :ni] = labels_i
        else:
            prepared_labels = []
            add_label_id = self.label2id[self.dum_label]
            for labels_i, mask in zip(labels, self.predict_mask_tensor):
                pre_labels = []
                cnt = 0
                n = len(labels_i)
                for idx, flag in enumerate(mask):
                    if flag:
                        pre_labels.append(labels_i[cnt])
                        cnt += 1
                    else:
                        if n == cnt:
                            pre_labels.append(O_id)
                        else:
                            pre_labels.append(add_label_id)
                prepared_labels.append(pre_labels)

        return torch.LongTensor(prepared_labels)

    def __getitem__(self, idx):
        idx = idx % self.n_data_
        d = {
            'ids'           : idx,
            'input_ids'     : self.input_ids_tensor[idx],
            'attention_mask': self.input_mask_tensor[idx],
            'mask'          : self.predict_mask_tensor[idx],
        }
        return d




class NeuralHiddenCrf(BaseTorchSeqModel):
    def __init__(self,
                 lr_worker: Optional[float] = 1e-3,
                 scaling: Optional[int] = 4,
                 model_name: Optional[str] = 'bert-base-uncased',
                 lr: Optional[float] = 2e-5,
                 lr_mv: Optional[float] = 2e-5,
                 l2: Optional[float] = 1e-6,
                 max_tokens: Optional[int] = 512,
                 batch_size: Optional[int] = 32,
                 batch_size_mv: Optional[int] = 32,
                 real_batch_size: Optional[int] = 32,
                 test_batch_size: Optional[int] = 128,
                 n_steps: Optional[int] = 10000,
                 use_crf: Optional[bool] = False,
                 fine_tune_layers: Optional[int] = -1,
                 lr_crf: Optional[float] = 5e-5,
                 lr_crf_mv: Optional[float] = 5e-5,
                 l2_crf: Optional[float] = 1e-8
                 ):
        super().__init__()

        self.pre_best_model = None
        self.pre_best_step = -1
        self.pre_best_metric_value = 0.0
        self.hyperparas = {
            'scaling'               : scaling,
            'lr_worker'             : lr_worker,
            'fine_tune_layers': fine_tune_layers,
            'model_name'      : model_name,
            'lr'              : lr,
            'lr_mv': lr_mv,
            'l2'              : l2,
            'max_tokens'      : max_tokens,
            'batch_size'      : batch_size,
            'batch_size_mv': batch_size_mv,
            'real_batch_size' : real_batch_size,
            'test_batch_size' : test_batch_size,
            'n_steps'         : n_steps,
            'use_crf'         : use_crf,
            'lr_crf'          : lr_crf,
            'lr_crf_mv': lr_crf_mv,
            'l2_crf'          : l2_crf,
        }
        print('self.hyperparas:', self.hyperparas)

        self.model = None
        # model_path = r"../bert-base-uncased/"
        # model_path = r"./bert-base-uncased/"
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def _init_valid_dataloader(self, dataset_valid: BaseSeqDataset) -> DataLoader:
        torch_dataset = BERTTorchSeqDataset(dataset_valid, self.tokenizer, 512, self.hyperparas['use_crf'])
        valid_dataloader = DataLoader(torch_dataset, batch_size=32, shuffle=False,
                                      collate_fn=collate_fn)
        return valid_dataloader



    def fit(self,
            dataset_train: BaseSeqDataset,
            test_data: BaseSeqDataset,
            pi: Optional[List[List[List]]] = None,
            ablation: Optional[str] = '0',
            y_train_crowd: Optional[List[List[List]]] = None,
            y_train_mv: Optional[List[List]] = None,
            dataset_valid: Optional[BaseSeqDataset] = None,
            y_valid: Optional[List[List]] = None,
            evaluation_step: Optional[int] = 50,
            metric: Optional[Union[str, Callable]] = 'f1_seq',
            strict: Optional[bool] = True,
            direction: Optional[str] = 'auto',
            patience_mv: Optional[int] = 100,
            patience: Optional[int] = 20,
            tolerance: Optional[float] = -1.0,
            device: Optional[torch.device] = None,
            verbose: Optional[bool] = True,
            result_path=None,
            **kwargs: Any):

        if not verbose:
            logger.setLevel(logging.ERROR)
        self._update_hyperparas(**kwargs)
        hyperparas = self.hyperparas
        accum_steps = 1
        n_steps = hyperparas['n_steps']


        torch_dataset = BERTTorchSeqDataset(dataset_train, self.tokenizer, self.hyperparas['max_tokens'],
                                            self.hyperparas['use_crf'], n_data=n_steps * hyperparas['batch_size_mv'])
        train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['batch_size_mv'], shuffle=True, collate_fn=collate_fn)



        worker_number = len(y_train_crowd[0][0])
        n, max_seq_len = torch_dataset.predict_mask_tensor.shape
        O_id = torch_dataset.label2id['O']
        y_train_crowd_pad = np.ones((n, max_seq_len, worker_number), dtype=int) * O_id
        for i, y_train_crowd_i in enumerate(y_train_crowd):
            ni = len(y_train_crowd_i)
            if ni > max_seq_len:
                y_train_crowd_pad[i, :len(y_train_crowd_i), :] = y_train_crowd_i[:max_seq_len, :]
            else:
                y_train_crowd_pad[i, :len(y_train_crowd_i), :] = y_train_crowd_i
        y_train_crowd = torch.LongTensor(y_train_crowd_pad)
        y_train_mv = torch_dataset.prepare_labels(y_train_mv)



        n_class = dataset_train.n_class
        model = BertSeqTagger(
            worker_number=worker_number,
            n_class=n_class,
            pi=pi,
            ablation=ablation,
            **hyperparas).to(device)
        self.model = model



        worker_params = []
        crf_param = []
        other_params = []
        print("model.named_parameters()", model.named_parameters())
        for name, para in model.named_parameters():
            if "worker_transitions" in name:
                print("worker_transitions: ", name)
                worker_params += [para]
            elif 'crf.transitions' in name:
                print("crf_transitions: ", name)
                crf_param += [para]
            else:
                other_params += [para]
        optimizer_grouped_parameters = [
            {"params": worker_params, "lr": hyperparas['lr_worker'], 'weight_decay': hyperparas['l2_crf']},
            {'params': other_params},
            {'params': crf_param, 'lr': hyperparas['lr_crf_mv'], 'weight_decay': hyperparas['l2_crf']},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparas['lr_mv'], weight_decay=hyperparas['l2'])


        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)
        valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, strict, direction, patience_mv, tolerance)

        history = {}
        last_step_log = {}
        try:
            print("Start pre-training for parameter initialization.")
            with trange(n_steps, desc=f"[FINETUNE] {hyperparas['model_name']} Tagger", unit="steps", disable=not verbose, ncols=150, position=0, leave=True) as pbar:
                cnt = 0
                step = 0
                model.train()
                optimizer.zero_grad()

                for batch in train_dataloader:
                    torch.cuda.empty_cache()
                    batch_idx = batch['ids'].to(device)
                    batch_label = y_train_mv[batch_idx].to(device)
                    loss = model.calculate_loss(batch, batch_label)
                    loss.backward()
                    cnt += 1

                    if cnt % accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        step += 1

                        if valid_flag and step % evaluation_step == 0:
                            metric_value, early_stop_flag, info, _ = self._valid_step(step, result_path)
                            if early_stop_flag:
                                logger.info(info)
                                break
                            history[step] = {
                                'loss'              : loss.item(),
                                f'val_{metric}'     : metric_value,
                                f'best_val_{metric}': self.best_metric_value,
                                'best_step'         : self.best_step,
                            }
                            last_step_log.update(history[step])

                        last_step_log['loss'] = loss.item()
                        pbar.update()
                        pbar.set_postfix(ordered_dict=last_step_log)

                        if step >= n_steps:
                            break


            self._finalize()
            model.train()



            ########################################################################################################
            worker_params = []
            crf_param = []
            other_params = []
            for name, para in model.named_parameters():
                if "worker_transitions" in name:
                    print("worker_transitions: ", name)
                    worker_params += [para]
                elif 'crf.transitions' in name:
                    print("crf_transitions: ", name)
                    crf_param += [para]
                else:
                    other_params += [para]
            optimizer_grouped_parameters = [
                {"params": worker_params, "lr": hyperparas['lr_worker'], 'weight_decay': hyperparas['l2_crf']},
                {'params': other_params},
                {'params': crf_param, 'lr': hyperparas['lr_crf'], 'weight_decay': hyperparas['l2_crf']},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=hyperparas['lr'], weight_decay=hyperparas['l2'])





            torch_dataset = BERTTorchSeqDataset(dataset_train, self.tokenizer, self.hyperparas['max_tokens'],
                                                self.hyperparas['use_crf'], n_data=n_steps * hyperparas['batch_size'])
            train_dataloader = DataLoader(torch_dataset, batch_size=hyperparas['batch_size'], shuffle=True,
                                          collate_fn=collate_fn)
            # Set up the learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=n_steps)
            valid_flag = self._init_valid_step(dataset_valid, y_valid, metric, strict, direction, patience,
                                               tolerance)

            self.best_model = None
            self.best_step = -1
            self.best_metric_value = 0.0
            all_val_loss, all_val_results = [], []


            print("Start formal training.")
            print("n_steps", n_steps)
            cnt = 0
            step = 0
            optimizer.zero_grad()
            for batch in train_dataloader:
                torch.cuda.empty_cache()
                batch_idx = batch['ids'].to(device)

                batch_label_crowd = y_train_crowd[batch_idx].to(device)
                loss = model.calculate_loss_crowd(batch, batch_label_crowd, worker_number)
                loss.backward()
                cnt += 1

                if cnt % accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                    if valid_flag and step % evaluation_step == 0:
                        metric_value, early_stop_flag, info, best_step_all_time = self._valid_step_2(step, result_path)
                        if early_stop_flag:
                            logger.info(info)
                            break

                        all_val_loss.append(round(loss.item(), 2))
                        all_val_results.append(round(metric_value, 2))

                        print('\n')
                        print("All_val_loss:", all_val_loss)
                        print("All_val_results:", all_val_results)
                        print('Best_step_all_time:', best_step_all_time)

                        history[step] = {
                            'loss'              : loss.item(),
                            f'val_{metric}'     : metric_value,
                            f'best_val_{metric}': self.best_metric_value,
                            'best_step'         : self.best_step,
                        }
                        last_step_log.update(history[step])

                    if step >= n_steps:
                        break


        except KeyboardInterrupt:
            logger.info(f'KeyboardInterrupt! do not terminate the process in case need to save the best model')

        print('\n')
        print("All_val_loss:", all_val_loss)
        print("All_val_results:", all_val_results)
        print('\n')


        np.save(os.path.join(result_path, 'all_val_loss.npy'), all_val_loss)
        np.save(os.path.join(result_path, 'all_val_results.npy'), all_val_results)


        self._finalize()
        return history, y_train_crowd




