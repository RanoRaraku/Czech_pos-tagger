import datetime

import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import StepLR

from Morpho import MorphoDataset
from transformer import Transformer

def eval_accuracy(model, dloader):
    """
    Returns accuracy.

    :dloader: torch.utils.data.DataLoader object
    """
    model.eval()
    total_samples, corr = 0, 0
    with torch.no_grad():
        for batch in dloader:
            words = batch["words"].to(model.device)
            tags = batch["tags"].to(model.device)
            words_num = batch["words_num"].to(model.device) - 1
            output_targets = tags[:, 1:]

            max_words_num = torch.max(words_num)
            mask = torch.arange(max_words_num, device=model.device).expand(
                len(words_num), max_words_num
            ) < words_num.unsqueeze(1)

            # Run inference
            y_hat = model(words, max_seq_len=max_words_num)
            corr += torch.sum(torch.argmax(y_hat[mask], dim=-1) == output_targets[mask])
            total_samples += torch.sum(words_num)

    return corr / total_samples


def train_epoch(
    model,
    train_dataloader,
    dev_dataloader,
    loss_fn,
    optim,
    lr_scheduler: Optional[Callable] = None,
    logger: Optional[Callable] = None,
):
    """
    :train_dataloader: a torch.utils.data.DataLoader object
    :dev_dataloader: a torch.utils.data.DataLoader object
    :loss_fn: a callable loss function
    :optim: a torch.optim object
    :lr_scheduler: a learning rate scheduler
    :logger: a callable logger (i.e wandb)
    """

    # TRAIN on train
    start_time = time.time()
    model.train()
    for batch in train_dataloader:
        encoder_inputs = batch["words"].to(model.device)
        tags = batch["tags"].to(model.device)
        words_num = batch["words_num"].to(model.device) - 1

        max_words_num = torch.max(words_num)
        mask = torch.arange(max_words_num, device=model.device).expand(
            len(words_num), max_words_num
        ) < words_num.unsqueeze(1)

        # Run inference
        decoder_inputs = tags[:, :-1]
        decoder_targets = tags[:, 1:]
        y_hat = model(encoder_inputs, decoder_inputs)
        loss = loss_fn(y_hat[mask], decoder_targets[mask])

        # Update params
        optim.zero_grad()
        loss.backward()
        optim.step()

        if logger is not None:
            logger.log({"train_loss": loss.item()})

    # EVAL on dev
    dev_samples, dev_corr, dev_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for batch in dev_dataloader:
            encoder_inputs = batch["words"].to(model.device)
            tags = batch["tags"].to(model.device)
            words_num = batch["words_num"].to(model.device) - 1

            max_words_num = torch.max(words_num)
            mask = torch.arange(max_words_num, device=model.device).expand(
                len(words_num), max_words_num
            ) < words_num.unsqueeze(1)

            # Run inference
            decoder_inputs = tags[:, :-1]
            decoder_targets = tags[:, 1:]
            y_hat = model(encoder_inputs, decoder_inputs)
            loss = loss_fn(y_hat[mask], decoder_targets[mask])

            dev_loss += loss.detach().item()
            dev_corr += torch.sum(
                torch.argmax(y_hat[mask], dim=-1) == decoder_targets[mask]
            )
            dev_samples += torch.sum(words_num)
    dev_acc = dev_corr / dev_samples
    dev_loss /= len(dev_dataloader)
    end_time = time.time()

    # Log
    model.epoch += 1
    if lr_scheduler is not None:
        lr_scheduler.step()
    if logger is not None:
        logger.log(
            {
                "epoch_time": end_time - start_time,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }
        )



morpho = MorphoDataset("czech_pdt")

args = {
    "batch_size": 128,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "czech_pdt",
    "model": "transformer",
    "heads": 8,
    "input_dropout": 0,
    "model_dim": 512,
    "keys_dim": 512,
    "values_dim": 512,
    "max_seq_len": morpho.max_length,
    "encoder_stack_size": 2,
    "decoder_stack_size": 2,
    "input_vocab_size": morpho.train.unique_words,
    "num_classes": morpho.train.unique_tags,
    "label_smoothing": 0.1,
    "packed_sequences": False,
    "characters": False,
}


model = Transformer(args)
optim = torch.optim.AdamW(model.parameters(), 0.00075)
loss_fn = nn.CrossEntropyLoss(label_smoothing=args["label_smoothing"])
train_dloader = morpho.train.to_dataloader(args["batch_size"], shuffle=True)
dev_dloader = morpho.dev.to_dataloader(args["batch_size"], shuffle=False)
scheduler = StepLR(optim, step_size=1, gamma=0.75)

wandb.login()
run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
wandb.init(project="tagger_competition", name=run_name, config=args)
for epoch in range(args["epochs"]):
    train_epoch(model, train_dloader, dev_dloader, loss_fn, optim, scheduler, None)
wandb.finish()
