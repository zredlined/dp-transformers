"""Train GPT2 model series with DP (w/ parameter-efficient approach LoRA when lora_dim > 0)"""

import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass, field

import datasets
import dp_transformers
import flwr as fl
import numpy as np
import torch
import transformers
from dp_transformers.grad_sample.lora import lora_layer
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora
from evaluate import load as load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, DataCollatorWithPadding
from transformers.training_args import ParallelMode

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda")
NUM_CLIENTS = 1
NUM_ROUNDS = 3


@dataclass
class ModelArguments:
    model_name: str = field(
        default="gpt2", metadata={"help": "Model name in HuggingFace, e.g. 'gpt2'"}
    )

    lora_dim: int = field(
        default=0, metadata={"help": "LoRA dimension; 0 means LoRA is disabled"}
    )

    sequence_len: int = field(default=128, metadata={"help": "Model sequence length"})

    lora_dropout: float = field(
        default=0.0, metadata={"help": "Dropout probability for LoRA layers"}
    )

    lora_alpha: int = field(default=32, metadata={"help": "LoRA attention alpha"})


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, model, dataset):
        self.net = model
        self.trainloader = dataset["train"]
        self.testloader = dataset["test"]

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, epochs=1)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return (
            float(loss),
            len(self.testloader),
            {"accuracy": float(accuracy), "loss": float(loss)},
        )


def train(model, dataset, epochs):
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_name)
    tokenizer.pad_token = -100  # Set a dummy pad token we don't use it anyway

    # Load collator
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(
        tokenizer
    )

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        privacy_args=privacy_args,
    )

    try:
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({"final_epsilon_prv": eps_prv, "final_epsilon_rdp": eps_rdp})


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Load dataset
    dataset = datasets.load_dataset("reddit", split="train[:500000]").train_test_split(
        0.02, seed=args.train.seed
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu:"
        f" {train_args.n_gpu}, distributed training:"
        f" {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    model = model.to(train_args.device)

    if args.model.lora_dim > 0:
        model = convert_gpt2_attention_to_lora(
            model,
            r=args.model.lora_dim,
            lora_alpha=args.model.lora_alpha,
            lora_dropout=args.model.lora_dropout,
            enable_lora=[True, False, True],
            merge_weights=False,
        )
        mark_only_lora_as_trainable(model)

    if train_args.local_rank == 0:
        logger.info(
            "Total number of parameters of the model:"
            f" {model.num_parameters(only_trainable=False)}"
        )
        logger.info(
            "Fine-tuned number of parameters of the model:"
            f" {model.num_parameters(only_trainable=True)}"
        )

    # Initialize model for simulation
    model = model.cuda()
    model.train()

    # Initialize flower client
    def client_fn(cid):
        return FederatedClient(model, dataset)

    # Start the simulation
    def weighted_average(metrics):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {
            "accuracy": sum(accuracies) / sum(examples),
            "loss": sum(losses) / sum(examples),
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1},
        ray_init_args={"log_to_driver": False, "num_cpus": 1, "num_gpus": 1},
    )


if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser(
        (
            dp_transformers.TrainingArguments,
            dp_transformers.PrivacyArguments,
            ModelArguments,
        )
    )
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))
