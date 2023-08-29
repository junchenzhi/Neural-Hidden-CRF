from abc import abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig



def log_sum_exp(smat):
    # status matrix (smat): (tagset_size, tagset_size)
    # @return (1, tagset_size)
    max_score = smat.max(dim=0, keepdim=True).values
    return (smat - max_score).exp().sum(axis=0, keepdim=True).log() + max_score


class BackBone(nn.Module):
    def __init__(self, n_class, binary_mode=False):
        if binary_mode:
            assert n_class == 2
            n_class = 1
        self.n_class = n_class
        super(BackBone, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.dummy_param.device

    def get_device(self):
        return self.dummy_param.device

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass


class BERTBackBone(BackBone):
    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, binary_mode=False):
        super(BERTBackBone, self).__init__(n_class=n_class, binary_mode=binary_mode)
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, num_labels=self.n_class, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

    @abstractmethod
    def forward(self, batch: Dict, return_features: Optional[bool] = False):
        pass






""" for sequence tagging """

class CRFTagger(BackBone):
    def __init__(self, worker_number, n_class, use_crf, pi, ablation, scaling):
        super(CRFTagger, self).__init__(n_class=n_class)
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(worker_number, n_class, pi, ablation, scaling)

    def calculate_loss_crowd(self, batch, batch_label_crowd, worker_number):
        device = self.get_device()
        outs = self.get_features(batch)

        mask = batch['mask'].to(device)
        batch_size, seq_len, _ = outs.shape
        batch_label_crowd = batch_label_crowd[:, :seq_len, :].to(device)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss_crowd(outs, mask, batch_label_crowd)
            total_loss = total_loss / (batch_size * worker_number)

        return total_loss


    def calculate_loss(self, batch, batch_label):
        device = self.get_device()
        outs = self.get_features(batch)
        mask = batch['mask'].to(device)
        batch_size, seq_len, _ = outs.shape
        batch_label = batch_label[:, :seq_len].to(device)

        if self.use_crf:
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            total_loss = total_loss / batch_size

        return total_loss


    def forward(self, batch):
        device = self.get_device()
        outs = self.get_features(batch)
        mask = batch['mask'].to(device)
        if self.use_crf:
            scores, tag_seq = self.crf(outs, mask)

        return tag_seq


    def forward_wrench_viterbi(self, batch):
        device = self.get_device()
        outs = self.get_features(batch)
        mask = batch['mask'].to(device)
        if self.use_crf:
            scores, tag_seq = self.crf.forward_wrench_viterbi(outs, mask)

        return tag_seq


    @abstractmethod
    def get_features(self, batch):
        pass






class BertSeqTagger(CRFTagger):
    """
    BERT for sequence tagging
    """

    def __init__(self,
                 worker_number,
                 n_class,
                 pi,
                 ablation,
                 model_name='bert-base-cased',
                 fine_tune_layers=-1,
                 use_crf=True,
                 **kwargs):
        super(BertSeqTagger, self).__init__(worker_number=worker_number, n_class=n_class, use_crf=use_crf, pi=pi, ablation=ablation,
                                            scaling=kwargs['scaling'])
        self.model_name = model_name
        config = AutoConfig.from_pretrained(self.model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(self.model_name, config=config)
        self.config = config

        if fine_tune_layers >= 0:
            for param in self.model.base_model.embeddings.parameters(): param.requires_grad = False
            if fine_tune_layers > 0:
                n_layers = len(self.model.base_model.encoder.layer)
                for layer in self.model.base_model.encoder.layer[:n_layers - fine_tune_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_crf = use_crf
        if self.use_crf:
            self.classifier = nn.Linear(config.hidden_size, n_class + 2)  # consider <START> and <END> token
        else:
            self.classifier = nn.Linear(config.hidden_size, n_class + 1)

    def get_features(self, batch):
        device = self.get_device()
        outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch['attention_mask'].to(device))
        outs = self.classifier(self.dropout(outputs.last_hidden_state))
        if self.use_crf:
            return outs
        else:
            return outs[:, :, :-1]







class CRF(BackBone):
    def __init__(self, worker_number, n_class, pi, ablation, scaling, batch_mode=True):
    # def __init__(self, worker_number, n_class, pi, scaling, batch_mode=False):
        super(CRF, self).__init__(n_class=n_class)
        # Matrix of transition parameters.  Entry i,j is the score of transitioning from i to j.
        self.n_class = n_class + 2
        self.worker_number = worker_number
        self.batch_mode = batch_mode
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.randn(self.n_class, self.n_class)
        self.START_TAG = -2
        self.STOP_TAG = -1

        init_transitions[:, self.START_TAG] = -1e5
        init_transitions[self.STOP_TAG, :] = -1e5
        self.transitions = nn.Parameter(init_transitions, requires_grad=True)

        self.pi = pi
        self.ablation = ablation

        worker_transitions = torch.randn(self.worker_number, self.n_class, self.n_class)
        for j in range(n_class):
            for k in range(n_class):
                worker_transitions.data[:, j, k] = 0.0
                worker_transitions.data[:, j, k] = worker_transitions.data[:, j, k] + scaling * self.pi[:, j, k]

        worker_transitions.data[:, :, self.START_TAG] = -10000.0
        worker_transitions.data[:, self.STOP_TAG, :] = -10000.0
        worker_transitions.data[:, self.START_TAG, :] = -10000.0
        worker_transitions.data[:, :, self.STOP_TAG] = -10000.0
        self.worker_transitions = nn.Parameter(worker_transitions, requires_grad=True)
        self.start_id = nn.Parameter(torch.LongTensor([self.START_TAG]), requires_grad=False)
        self.stop_id = nn.Parameter(torch.LongTensor([self.STOP_TAG]), requires_grad=False)


    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    def _score_sentence(self, feats, tags, transitions=None):
        # Gives the score of a provided tag sequence
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * n_class

        if transitions is None:
            transitions = self.transitions

        pad_start_tags = torch.cat([self.start_id, tags])
        pad_stop_tags = torch.cat([tags, self.stop_id])

        r = torch.arange(feats.size(0))
        score = torch.sum(transitions[pad_start_tags, pad_stop_tags]) + torch.sum(feats[r, tags])
        return score


    def _score_sentence_batch(self, feats, tags, mask, transitions=None):
        # Gives the score of a provided tag sequence
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * n_class
        if transitions is None:
            transitions = self.transitions

        batch_size = tags.size(0)
        seq_len = mask.long().sum(1)
        r_batch = torch.arange(batch_size)

        pad_start_tags = torch.cat([self.start_id.expand(batch_size, 1), tags], -1)
        pad_stop_tags = torch.cat([tags, self.stop_id.expand(batch_size, 1)], -1)

        pad_stop_tags[r_batch, seq_len] = self.stop_id
        t = transitions[pad_start_tags, pad_stop_tags]
        t_score = torch.sum(t.cumsum(1)[r_batch, seq_len])

        f_score = torch.sum(torch.gather(feats, -1, tags.unsqueeze(2)).squeeze(2).masked_select(mask.bool()))

        score = t_score + f_score
        return score


    def _forward_alg(self, feats, transitions=None):
        # calculate in log domain
        # feats is len(sentence) * n_class
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        alpha = torch.full((1, self.n_class), -10000.0, device=device)
        alpha[0][self.START_TAG] = 0.0
        for feat in feats:
            alpha = torch.logsumexp(alpha.T + feat.unsqueeze(0) + transitions, dim=0, keepdim=True)
        return torch.logsumexp(alpha.T + 0 + transitions[:, [self.STOP_TAG]], dim=0)[0]


    def _forward_alg_batch(self, feats, mask, transitions=None):
        # calculate in log domain
        # feats is len(sentence) * n_class
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        batch_size, max_seq_len, target_size = feats.shape
        alpha = torch.full((batch_size, 1, target_size), -10000.0, device=device)
        alpha[:, 0, self.START_TAG] = 0.0
        mask = mask.bool()

        for i in range(max_seq_len):
            feat = feats[:, i, :]
            mask_i = mask[:, i]
            alpha = torch.where(mask_i.view(-1, 1, 1),
                                torch.logsumexp(alpha.transpose(1, 2) + feat.unsqueeze(1) + transitions, dim=1,
                                                keepdim=True), alpha)

        last = torch.logsumexp(alpha.transpose(1, 2) + 0 + transitions[:, [self.STOP_TAG]], dim=1)
        score = torch.sum(last)
        return score


    def _score_sentence_crowd(self, feats, tags):
        device = self.get_device()
        alpha = torch.full((1, self.n_class), -10000.0, device=device)
        alpha[0][self.START_TAG] = 0.0

        for i, feat in enumerate(feats):
            beta = torch.full((1, self.n_class), 0.0, device=device)
            for worker_id in range(self.worker_number):
                tag = tags[i, worker_id]
                beta += self.worker_transitions[worker_id, :, tag.item()].unsqueeze(0)
            alpha = torch.logsumexp(alpha.T + feat.unsqueeze(0) + self.transitions + beta, dim=0, keepdim=True)

        return torch.logsumexp(alpha.T + 0 + self.transitions[:, [self.STOP_TAG]], dim=0)[0]


    def _forward_alg_crowd(self, feats, factors_weaksupervision):
        device = self.get_device()
        alpha = torch.full((1, self.n_class), -10000.0, device=device)
        alpha[0][self.START_TAG] = 0.0

        for i, feat in enumerate(feats):
            alpha = torch.logsumexp(alpha.T + feat.unsqueeze(0) + self.transitions + factors_weaksupervision, dim=0, keepdim=True)

        return torch.logsumexp(alpha.T + 0 + self.transitions[:, [self.STOP_TAG]], dim=0)[0]


    def viterbi_decode_batch(self, feats, mask, transitions=None):
        if transitions is None:
            transitions = self.transitions

        device = self.get_device()
        batch_size, max_seq_len, target_size = feats.shape
        backtrace = torch.zeros((batch_size, max_seq_len, target_size)).long()
        alpha = torch.full((batch_size, 1, target_size), -10000.0, device=device)
        alpha[:, 0, self.START_TAG] = 0.0
        mask = mask.bool()

        for i in range(max_seq_len):
            feat = feats[:, i, :]
            mask_i = mask[:, i]
            smat = (alpha.transpose(1, 2) + feat.unsqueeze(1) + transitions)  # (n_class, n_class)
            backtrace[:, i, :] = smat.argmax(1)
            alpha_2, _ = smat.max(dim=1)
            alpha_2 = alpha_2.unsqueeze(1)
            alpha = torch.where(mask_i.view(-1, 1, 1), alpha_2, alpha)
            # alpha = torch.where(mask_i.view(-1, 1, 1), torch.logsumexp(smat, dim=1, keepdim=True), alpha)

        # backtrack
        smat = alpha.transpose(1, 2) + 0 + transitions[:, [self.STOP_TAG]]
        best_tag_ids = smat.argmax(1).long()

        seq_len = mask.long().sum(1)
        best_paths = []
        for backtrace_i, best_tag_id, l in zip(backtrace, best_tag_ids, seq_len):
            best_path = [best_tag_id.item()]
            for bptrs_t in reversed(backtrace_i[1:l]):  # ignore START_TAG
                best_tag_id = bptrs_t[best_tag_id].item()
                best_path.append(best_tag_id)
            best_paths.append(best_path[::-1])
        return torch.logsumexp(smat, dim=1).squeeze().tolist(), best_paths

    ########################################################################################################
    ########################################################################################################
    ########################################################################################################


    def neg_log_likelihood_loss(self, feats, mask, tags, transitions=None):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.n_class
        if self.batch_mode:
            nll_loss = self._forward_alg_batch(feats, mask, transitions) - self._score_sentence_batch(feats, tags, mask,
                                                                                                      transitions)
        else:
            nll_loss = 0.0
            batch_size = len(feats)
            for i in range(batch_size):
                length = mask[i].long().sum()
                feat_i = feats[i][:length]
                tags_i = tags[i][:length]
                forward_score = self._forward_alg(feat_i, transitions)
                gold_score = self._score_sentence(feat_i, tags_i, transitions)
                nll_loss += forward_score - gold_score
        return nll_loss


    def neg_log_likelihood_loss_crowd(self, feats, mask, tags):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        # factors = torch.sum(torch.exp(worker_transitions), dim=2)

        all_loss = 0
        factors_weaksupervision = self._get_factors_weaksupervision()

        for i in range(len(feats)):
            length = mask[i].long().sum()
            feat_i = feats[i][:length]
            tags_i = tags[i][:length]

            gold_score = self._score_sentence_crowd(feat_i, tags_i)
            forward_score = self._forward_alg_crowd(feat_i, factors_weaksupervision)
            all_loss += forward_score - gold_score

        return all_loss


    def forward(self, feats, mask):
        tags = []
        scores = []
        score, tag_seq_2 = self.viterbi_decode_batch(feats, mask)
        tags.append(tag_seq_2)

        return scores, tag_seq_2


    def argmax(self, vec):
        _, idx = torch.max(vec, 1)
        return idx.item()


    def _get_factors_weaksupervision(self):
        device = self.get_device()
        alpha = torch.full((1, self.n_class), 0.0, device=device)
        alpha[0][self.START_TAG] = -10000.0
        alpha[0][self.STOP_TAG] = -10000.0

        for i in range(self.n_class-2):
            beta = torch.full((1, self.n_class), 0.0, device=device)
            for j in range(self.worker_number):
                beta = torch.logsumexp(beta.T + self.worker_transitions[j][i].unsqueeze(0), dim=0, keepdim=True)
            alpha[0][i] = torch.logsumexp(beta.T, dim=0)[0]

        return alpha





