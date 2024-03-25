import torch


class constant(object):
    def __init__(self):
        # 以下来自CodeReviewer
        self.epoch = 12
        self.num_train_epochs = self.epoch
        self.max_grad_norm=1.0
        self.train_batch_size = 20
        self.adam_epsilon = 1e-8

        self.learning_rate = 3e-4

        self.gradient_accumulation_steps = 1

        self.output_dir = "./saved_models"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name_or_path= "microsoft / codereviewer"


        self.data_path = r"./data/code_hc.parquet"
        self.label_path = r"./data/zzx.csv"

        self.max_source_length=512
        self.max_target_length = 128
        self.mask_rate=0.15
        self.save_steps=3600
        self.log_steps=100
        self.train_steps=120000
        self.seed=2233
        self.load_model_path=None

        # 以下来自CodeBERT
        self.local_rank = -1
        self.start_step = 0
        self.start_epoch = 0
        self.eval_batch_size = 10
        self.weight_decay = 0.0
        self.fp16 = False
        self.max_grad_norm = 1.0
        self.evaluate_during_training = True
        self.fp16_opt_level = "01"

