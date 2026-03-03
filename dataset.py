import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import GPTTokenizer


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = self.tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(
    text, batch_size=4, max_length=256, stride=128, shuffle=True, tokenizer=None
):
    if tokenizer is None:
        tokenizer = GPTTokenizer()
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class HuggingFaceGPTDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length, stride, max_samples=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.input_ids = []
        self.target_ids = []

        samples = (
            list(hf_dataset) if max_samples is None else list(hf_dataset)[:max_samples]
        )

        for example in samples:
            text = example["text"]
            token_ids = self.tokenizer.encode(text)

            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i : i + max_length]
                target_chunk = token_ids[i + 1 : i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_from_huggingface(
    hf_dataset,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    max_samples=None,
):
    dataset = HuggingFaceGPTDataset(
        hf_dataset, tokenizer, max_length, stride, max_samples
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    import os

    file_path = "the-verdict.txt"

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print(f"Loaded the verdict , total chartacters : {len(raw_text)}")

        dataloader = create_dataloader(raw_text, batch_size=2, max_length=10, stride=5)

        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)

        tokenizer = GPTTokenizer()
        print("Batch 1, Example 1:")
        print(f"Input text: '{tokenizer.decode(inputs[0].tolist())}'")
        print(f"Input tokens: {inputs[0].tolist()}")
        print("-" * 30)
        print(f"Target text: '{tokenizer.decode(targets[0].tolist())}'")
        print(f"Target tokens: {targets[0].tolist()}")
    else:
        print(f"Could not find {file_path}. Please make sure it exists.")
