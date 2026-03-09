import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tokenizer import GPTTokenizer
from tqdm import tqdm


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
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length,
        stride,
        max_samples=None,
        streaming=False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.streaming = streaming

        if streaming:
            self.hf_dataset = hf_dataset
            self.max_samples = max_samples
            self.total_samples = max_samples if max_samples else float("inf")
        else:
            if max_samples is not None:
                self.hf_dataset = list(hf_dataset[:max_samples])
            else:
                self.hf_dataset = list(hf_dataset)
            self.total_samples = len(self.hf_dataset)
            self._build_sequences()

    def _build_sequences(self):
        self.sequences = []
        for example in tqdm(self.hf_dataset, desc="Tokenizing"):
            # Handle depending on if huggingface dataset returns dict or raw string when listed
            text = example["text"] if isinstance(example, dict) else example
            token_ids = self.tokenizer.encode(text)

            for i in range(0, len(token_ids) - self.max_length, self.stride):
                input_chunk = token_ids[i : i + self.max_length]
                target_chunk = token_ids[i + 1 : i + self.max_length + 1]
                self.sequences.append(
                    (torch.tensor(input_chunk), torch.tensor(target_chunk))
                )

    def __len__(self):
        return int(self.total_samples)

    def __iter__(self):
        count = 0
        for example in self.hf_dataset:
            if self.max_samples and count >= self.max_samples:
                break

            text = example["text"]
            token_ids = self.tokenizer.encode(text)

            for i in range(0, len(token_ids) - self.max_length, self.stride):
                if self.max_samples and count >= self.max_samples:
                    break

                input_chunk = token_ids[i : i + self.max_length]
                target_chunk = token_ids[i + 1 : i + self.max_length + 1]

                yield torch.tensor(input_chunk), torch.tensor(target_chunk)
                count += 1

    def __getitem__(self, index):
        if self.streaming:
            raise NotImplementedError("Use iterator for streaming mode")
        return self.sequences[index]


class StreamingDataLoader:
    def __init__(
        self, hf_dataset, tokenizer, batch_size, max_length, stride, max_samples=None
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.max_samples = max_samples

    def __iter__(self):
        batch_inputs = []
        batch_targets = []
        count = 0

        for example in self.hf_dataset:
            if self.max_samples and count >= self.max_samples:
                break

            text = example["text"]
            token_ids = self.tokenizer.encode(text)

            for i in range(0, len(token_ids) - self.max_length, self.stride):
                if self.max_samples and count >= self.max_samples:
                    break

                input_chunk = token_ids[i : i + self.max_length]
                target_chunk = token_ids[i + 1 : i + self.max_length + 1]

                batch_inputs.append(torch.tensor(input_chunk))
                batch_targets.append(torch.tensor(target_chunk))
                count += 1

                if len(batch_inputs) >= self.batch_size:
                    yield torch.stack(batch_inputs), torch.stack(batch_targets)
                    batch_inputs = []
                    batch_targets = []

        if batch_inputs:
            yield torch.stack(batch_inputs), torch.stack(batch_targets)

    def __len__(self):
        return None  # Unknown length in streaming mode


def create_dataloader_from_huggingface(
    hf_dataset,
    tokenizer,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    max_samples=None,
    streaming=False,
    is_distributed=False,
):
    if streaming:
        return StreamingDataLoader(
            hf_dataset=hf_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            max_samples=max_samples,
        )

    dataset = HuggingFaceGPTDataset(
        hf_dataset, tokenizer, max_length, stride, max_samples
    )
    sampler = DistributedSampler(dataset) if is_distributed else None
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
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
