from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

from datasets import load_dataset


def main():
    # ds = load_dataset("wmt/wmt14", "fr-en")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train+test+validation")

    def text_iter(batch_size: int = 1024):
        tok_dataset = ds.select_columns("text")
        for batch in tok_dataset.iter(batch_size):
            yield batch["text"]

    tokenizer = Tokenizer(WordPiece())
    trainer = WordPieceTrainer(
        vocab_size=int(30e3),
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )
    tokenizer.train_from_iterator(text_iter(), trainer, len(ds))

    tokenizer.save("tokenizer")


if __name__ == "__main__":
    main()
