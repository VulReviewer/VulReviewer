import os
from itertools import chain

import torch.nn as nn
import torch

# from peft import PrefixTuningConfig, TaskType, get_peft_model
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import numpy as np
# from utils import MyTokenizer
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer, AutoModel,
)
import logging

from VIB import VIB

class EarlyStopping:
    """Early stops the training if validation f1 doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : model save path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_max = np.Inf
        self.delta = delta

    def __call__(self, f1, model):

        score = f1

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1, model)
            self.counter = 0

    def save_checkpoint(self, f1, model):
        '''Saves model when f1 increase.'''
        if self.verbose:
            print(f'Validation f1 increased ({self.f1_max:.6f} --> {f1:.6f}).  Saving model ...')
        self.f1_max = f1
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# save best model
        self.val_loss_min = f1

# from CodeReviewer.Method.main1 import prepare_VIB
def prepare_VIB(config):
    config.deterministic = False
    config.ib = True
    config.kl_annealing = "linear"
    config.activation = "relu"
    config.ib_dim = 384
    config.hidden_dim = (768 + config.ib_dim) // 2

    config.beta = 1e-05
    config.sample_size = 5
    config.is_VIB = True
    return config


logger = logging.getLogger(__name__)


class ReviewerModel_func(nn.Module):

    def __init__(self, config, base):
        super(ReviewerModel_func, self).__init__()
        # super().__init__(config)
        hidden_size = 768
        self.VIB = config.is_VIB
        self.mtl = config.mtl
        # self.new_loss = new_loss
        self.encoder= base.encoder
        # self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 7, device= "cuda"))
        # self.cls_head = nn.Linear(self.config.d_model, 3, bias=True)
        if self.VIB:
            self.VIB_C = VIB(config, num_labels=3)
            self.VIB_I = VIB(config, num_labels=3)
            self.VIB_A = VIB(config, num_labels=3)

            self.VIB_AV = VIB(config, num_labels=3)
            self.VIB_AC = VIB(config, num_labels=3)
            self.VIB_AU = VIB(config, num_labels=2)

            self.VIB_severity = VIB(config, 3)
        else:
            self.classifier_C = nn.Linear(hidden_size, 3)
            self.classifier_I = nn.Linear(hidden_size, 3)
            self.classifier_A = nn.Linear(hidden_size, 3)
            self.classifier_AV = nn.Linear(hidden_size, 3)
            self.classifier_AC = nn.Linear(hidden_size, 3)
            self.classifier_AU = nn.Linear(hidden_size, 2)
            self.classifier_severity = nn.Linear(hidden_size, 3)
            self.dropout = nn.Dropout(0.3)

        # self.init()

    def init(self):
        nn.init.xavier_uniform_(self.encoder.lm_head.weight)
        factor = self.config.initializer_factor

    def forward(
            self, *argv, **kwargs
    ):
        return self.cls(
            labels=kwargs["labels"],
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            epoch=kwargs["epoch"],
        )

    def cls(
            self,
            input_ids,
            attention_mask,
            labels,
            epoch=1
    ):
        # print(self.encoder)
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_attentions=False,
            # return_dict=False
        )
        criterion = nn.CrossEntropyLoss()
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]

        if self.VIB:
            cvss_C_Entity = self.VIB_C(first_hidden, labels=labels["true_C"], epoch=epoch)
            cvss_I_Entity = self.VIB_I(first_hidden, labels=labels["true_I"], epoch=epoch)
            cvss_A_Entity = self.VIB_A(first_hidden, labels=labels["true_A"], epoch=epoch)
            cvss_AV_Entity = self.VIB_AV(first_hidden, labels=labels["true_AV"], epoch=epoch)
            cvss_AC_Entity = self.VIB_AC(first_hidden, labels=labels["true_AC"], epoch=epoch)
            cvss_AU_Entity = self.VIB_AU(first_hidden, labels=labels["true_AU"], epoch=epoch)
            cvss_severity_Entity = self.VIB_severity(first_hidden, labels=labels["true_severity"], epoch=epoch)

            # total_loss = self.new_loss(cvss_C_Entity["loss"]["loss"], cvss_I_Entity["loss"]["loss"],
            #                      cvss_A_Entity["loss"]["loss"], cvss_AV_Entity["loss"]["loss"],
            #                      cvss_AC_Entity["loss"]["loss"], cvss_AU_Entity["loss"]["loss"],
            #                      cvss_severity_Entity["loss"]["loss"])

            # total_loss = cvss_C_Entity["loss"]["loss"] + cvss_I_Entity["loss"]["loss"] + cvss_A_Entity["loss"]["loss"] \
            #              + cvss_AV_Entity["loss"]["loss"] + cvss_AC_Entity["loss"]["loss"] + cvss_AU_Entity["loss"][
            #                  "loss"] \
            #              + cvss_severity_Entity["loss"]["loss"]
            # total_loss = self.get_loss_list(cvss_AC_Entity, cvss_AU_Entity, cvss_AV_Entity, cvss_A_Entity,
            #                                 cvss_C_Entity, cvss_I_Entity, cvss_severity_Entity)
            if self.mtl:
                total_loss = torch.stack([cvss_C_Entity["loss"]["loss"], cvss_I_Entity["loss"]["loss"], cvss_A_Entity["loss"]["loss"],
                                      cvss_AV_Entity["loss"]["loss"], cvss_AC_Entity["loss"]["loss"], cvss_AU_Entity["loss"]["loss"],
                                      cvss_severity_Entity["loss"]["loss"]])
            else:
                total_loss = (cvss_C_Entity["loss"]["loss"]+cvss_I_Entity["loss"]["loss"]+ cvss_A_Entity["loss"]["loss"]+ cvss_AV_Entity["loss"]["loss"]+ cvss_AC_Entity["loss"]["loss"]+
                              cvss_AU_Entity["loss"]["loss"]+cvss_severity_Entity["loss"]["loss"])
            # total_loss = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()

            cvss_C = cvss_C_Entity["logits"]
            cvss_I = cvss_I_Entity["logits"]
            cvss_A = cvss_A_Entity["logits"]
            cvss_AV = cvss_AV_Entity["logits"]
            cvss_AC = cvss_AC_Entity["logits"]
            cvss_AU = cvss_AU_Entity["logits"]
            cvss_severity = cvss_severity_Entity["logits"]
        else:
            # x = self.dropout(first_hidden)
            x = first_hidden
            cvss_C = self.classifier_C(x)
            cvss_I = self.classifier_I(x)
            cvss_A = self.classifier_A(x)
            cvss_AV = self.classifier_AV(x)
            cvss_AC = self.classifier_AC(x)
            cvss_AU = self.classifier_AU(x)
            cvss_severity = self.classifier_severity(x)

            loss_C = criterion(cvss_C, labels["true_C"])
            loss_I = criterion(cvss_I, labels["true_I"])
            loss_A = criterion(cvss_A, labels["true_A"])
            loss_AV = criterion(cvss_AV, labels["true_AV"])
            loss_AC = criterion(cvss_AC, labels["true_AC"])
            loss_AU = criterion(cvss_AU, labels["true_AU"])
            loss_severity = criterion(cvss_severity, labels["true_severity"])
            if self.mtl:
                total_loss = torch.stack([loss_C, loss_I, loss_A, loss_AV, loss_AC, loss_AU, loss_severity])
            else:
                total_loss = (loss_C + loss_I + loss_A + loss_AV + loss_AC + loss_AU + loss_severity)
        return cvss_C, cvss_I, cvss_A, cvss_AV, cvss_AC, cvss_AU, cvss_severity, total_loss,first_hidden  # x is bert output(768,)

    def get_loss_list(self, cvss_AC_Entity, cvss_AU_Entity, cvss_AV_Entity, cvss_A_Entity, cvss_C_Entity, cvss_I_Entity,
                      cvss_severity_Entity):
        total_loss = []
        total_loss.append(cvss_C_Entity["loss"]["loss"])
        total_loss.append(cvss_I_Entity["loss"]["loss"])
        total_loss.append(cvss_A_Entity["loss"]["loss"])
        total_loss.append(cvss_AV_Entity["loss"]["loss"])
        total_loss.append(cvss_AC_Entity["loss"]["loss"])
        total_loss.append(cvss_AU_Entity["loss"]["loss"])
        total_loss.append(cvss_severity_Entity["loss"]["loss"])
        return total_loss

    def shared_parameters(self):
        return (p for n, p in self.encoder.named_parameters())
    def task_specific_parameters(self):
        # all_parameters = []
        if self.VIB:
            return chain(
                self.VIB_C.parameters(),
                self.VIB_I.parameters(),
                self.VIB_A.parameters(),
                self.VIB_AV.parameters(),
                self.VIB_AC.parameters(),
                self.VIB_AU.parameters(),
                self.VIB_severity.parameters()
            )
        else:
            return chain(
                self.classifier_C.parameters(),
                self.classifier_I.parameters(),
                self.classifier_A.parameters(),
                self.classifier_AV.parameters(),
                self.classifier_AC.parameters(),
                self.classifier_AU.parameters(),
                self.classifier_severity.parameters()
            )

class ReviewerModel_commit(nn.Module):

    def __init__(self, config, base):
        super(ReviewerModel_commit, self).__init__()
        # super().__init__(config)
        hidden_size = 768
        self.VIB = config.is_VIB
        self.mtl = config.mtl
        # self.new_loss = new_loss
        # self.encoder= base.encoder
        self.encoder = RobertaModel.from_pretrained(r"D:\PLBART_MOBERT\MOBERT\codebert")

        if self.VIB:
            self.VIB_C = VIB(config, num_labels=3)
            self.VIB_I = VIB(config, num_labels=3)
            self.VIB_A = VIB(config, num_labels=3)

            self.VIB_AV = VIB(config, num_labels=2)
            self.VIB_AC = VIB(config, num_labels=3)
            self.VIB_AU = VIB(config, num_labels=2)

            self.VIB_severity = VIB(config, 3)
        else:
            self.classifier_C = nn.Linear(hidden_size, 3)
            self.classifier_I = nn.Linear(hidden_size, 3)
            self.classifier_A = nn.Linear(hidden_size, 3)
            self.classifier_AV = nn.Linear(hidden_size, 2)
            self.classifier_AC = nn.Linear(hidden_size, 3)
            self.classifier_AU = nn.Linear(hidden_size, 2)
            self.classifier_severity = nn.Linear(hidden_size, 3)
            self.dropout = nn.Dropout(0.3)

        # self.init()

    def init(self):
        nn.init.xavier_uniform_(self.encoder.lm_head.weight)
        factor = self.config.initializer_factor

    def forward(
            self, *argv, **kwargs
    ):
        return self.cls(
            labels=kwargs["labels"],
            input_ids=kwargs["input_ids"],
            attention_mask=kwargs["attention_mask"],
            epoch=kwargs["epoch"],
        )

    def cls(
            self,
            input_ids,
            attention_mask,
            labels,
            epoch=1
    ):
        # print(self.encoder)
        encoder_outputs = self.encoder( \
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_attentions=False,
            # return_dict=False
        )
        criterion = nn.CrossEntropyLoss()
        hidden_states = encoder_outputs[0]
        first_hidden = hidden_states[:, 0, :]

        if self.VIB:
            cvss_C_Entity = self.VIB_C(first_hidden, labels=labels["true_C"], epoch=epoch)
            cvss_I_Entity = self.VIB_I(first_hidden, labels=labels["true_I"], epoch=epoch)
            cvss_A_Entity = self.VIB_A(first_hidden, labels=labels["true_A"], epoch=epoch)
            cvss_AV_Entity = self.VIB_AV(first_hidden, labels=labels["true_AV"], epoch=epoch)
            cvss_AC_Entity = self.VIB_AC(first_hidden, labels=labels["true_AC"], epoch=epoch)
            cvss_AU_Entity = self.VIB_AU(first_hidden, labels=labels["true_AU"], epoch=epoch)
            cvss_severity_Entity = self.VIB_severity(first_hidden, labels=labels["true_severity"], epoch=epoch)

            # total_loss = self.new_loss(cvss_C_Entity["loss"]["loss"], cvss_I_Entity["loss"]["loss"],
            #                      cvss_A_Entity["loss"]["loss"], cvss_AV_Entity["loss"]["loss"],
            #                      cvss_AC_Entity["loss"]["loss"], cvss_AU_Entity["loss"]["loss"],
            #                      cvss_severity_Entity["loss"]["loss"])

            # total_loss = cvss_C_Entity["loss"]["loss"] + cvss_I_Entity["loss"]["loss"] + cvss_A_Entity["loss"]["loss"] \
            #              + cvss_AV_Entity["loss"]["loss"] + cvss_AC_Entity["loss"]["loss"] + cvss_AU_Entity["loss"][
            #                  "loss"] \
            #              + cvss_severity_Entity["loss"]["loss"]
            # total_loss = self.get_loss_list(cvss_AC_Entity, cvss_AU_Entity, cvss_AV_Entity, cvss_A_Entity,
            #                                 cvss_C_Entity, cvss_I_Entity, cvss_severity_Entity)
            if self.mtl:
                total_loss = torch.stack([cvss_C_Entity["loss"]["loss"], cvss_I_Entity["loss"]["loss"], cvss_A_Entity["loss"]["loss"],
                                      cvss_AV_Entity["loss"]["loss"], cvss_AC_Entity["loss"]["loss"], cvss_AU_Entity["loss"]["loss"],
                                      cvss_severity_Entity["loss"]["loss"]])
            else:
                total_loss = (cvss_C_Entity["loss"]["loss"]+cvss_I_Entity["loss"]["loss"]+ cvss_A_Entity["loss"]["loss"]+ cvss_AV_Entity["loss"]["loss"]+ cvss_AC_Entity["loss"]["loss"]+
                              cvss_AU_Entity["loss"]["loss"]+cvss_severity_Entity["loss"]["loss"])
            # total_loss = (losses/(2*self.loss_scale.exp())+self.loss_scale/2).sum()

            cvss_C = cvss_C_Entity["logits"]
            cvss_I = cvss_I_Entity["logits"]
            cvss_A = cvss_A_Entity["logits"]
            cvss_AV = cvss_AV_Entity["logits"]
            cvss_AC = cvss_AC_Entity["logits"]
            cvss_AU = cvss_AU_Entity["logits"]
            cvss_severity = cvss_severity_Entity["logits"]
        else:
            # x = self.dropout(first_hidden)
            x = first_hidden
            cvss_C = self.classifier_C(x)
            cvss_I = self.classifier_I(x)
            cvss_A = self.classifier_A(x)
            cvss_AV = self.classifier_AV(x)
            cvss_AC = self.classifier_AC(x)
            cvss_AU = self.classifier_AU(x)
            cvss_severity = self.classifier_severity(x)

            loss_C = criterion(cvss_C, labels["true_C"])
            loss_I = criterion(cvss_I, labels["true_I"])
            loss_A = criterion(cvss_A, labels["true_A"])
            loss_AV = criterion(cvss_AV, labels["true_AV"])
            loss_AC = criterion(cvss_AC, labels["true_AC"])
            loss_AU = criterion(cvss_AU, labels["true_AU"])
            loss_severity = criterion(cvss_severity, labels["true_severity"])
            if self.mtl:
                total_loss = torch.stack([loss_C, loss_I, loss_A, loss_AV, loss_AC, loss_AU, loss_severity])
            else:
                total_loss = (loss_C + loss_I + loss_A + loss_AV + loss_AC + loss_AU + loss_severity)
        return cvss_C, cvss_I, cvss_A, cvss_AV, cvss_AC, cvss_AU, cvss_severity, total_loss,first_hidden  # x is bert output(768,)

    def get_loss_list(self, cvss_AC_Entity, cvss_AU_Entity, cvss_AV_Entity, cvss_A_Entity, cvss_C_Entity, cvss_I_Entity,
                      cvss_severity_Entity):
        total_loss = []
        total_loss.append(cvss_C_Entity["loss"]["loss"])
        total_loss.append(cvss_I_Entity["loss"]["loss"])
        total_loss.append(cvss_A_Entity["loss"]["loss"])
        total_loss.append(cvss_AV_Entity["loss"]["loss"])
        total_loss.append(cvss_AC_Entity["loss"]["loss"])
        total_loss.append(cvss_AU_Entity["loss"]["loss"])
        total_loss.append(cvss_severity_Entity["loss"]["loss"])
        return total_loss

    def shared_parameters(self):

        return (p for n, p in self.encoder.encoder.named_parameters() if "pooler" not in n )
        # return a
    def task_specific_parameters(self):
        # all_parameters = []
        if self.VIB:
            return chain(
                self.VIB_C.parameters(),
                self.VIB_I.parameters(),
                self.VIB_A.parameters(),
                self.VIB_AV.parameters(),
                self.VIB_AC.parameters(),
                self.VIB_AU.parameters(),
                self.VIB_severity.parameters()
            )
        else:
            return chain(
                self.classifier_C.parameters(),
                self.classifier_I.parameters(),
                self.classifier_A.parameters(),
                self.classifier_AV.parameters(),
                self.classifier_AC.parameters(),
                self.classifier_AU.parameters(),
                self.classifier_severity.parameters()
            )

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e6))


def build_or_load_gen_model(args, VIB=False):

    config_class, tokenizer_class = T5Config, RobertaTokenizer

    config = config_class.from_pretrained(args.model_name_or_path)
    # torch.load()
    codeReview_base = AutoModel.from_pretrained(args.model_name_or_path,config=config)
    if VIB:
        config = prepare_VIB(config)
    else:
        config.is_VIB = False
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    # model = model_class.from_pretrained(args.model_name_or_path, new_loss=loss, config=config)
    if args.mtl ==True:
        config.mtl = True
    else:
        config.mtl = False

    if args.ablation =='function':
        model = ReviewerModel_func(base=codeReview_base, config=config)
    elif args.ablation =='commit':
        model = ReviewerModel_commit(base=codeReview_base, config=config)
    # tokenizer.special_dict = {
    #     f"<e{i}>": tokenizer.get_vocab()[f"<e{i}>"] for i in range(99, -1, -1)
    # }
    #
    # tokenizer.mask_id = tokenizer.get_vocab()["<mask>"]
    # tokenizer.bos_id = tokenizer.get_vocab()["<s>"]
    # tokenizer.pad_id = tokenizer.get_vocab()["<pad>"]
    # tokenizer.eos_id = tokenizer.get_vocab()["</s>"]
    # tokenizer.msg_id = tokenizer.get_vocab()["<msg>"]
    # tokenizer.keep_id = tokenizer.get_vocab()["<keep>"]
    # tokenizer.add_id = tokenizer.get_vocab()["<add>"]
    # tokenizer.del_id = tokenizer.get_vocab()["<del>"]
    # tokenizer.start_id = tokenizer.get_vocab()["<start>"]
    # tokenizer.end_id = tokenizer.get_vocab()["<end>"]

    print(get_model_size(model))
    # print(args.model_name_or_path)

    if args.load_model_path is not None:
        model_path = os.path.join(args.load_model_path, "pytorch_model.bin")
        logger.info("Reload model from {}".format(model_path))
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError:
            saved = model.cls_head
            model.cls_head = None
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.cls_head = saved
        model.to(args.device)

    return config, model, 0


