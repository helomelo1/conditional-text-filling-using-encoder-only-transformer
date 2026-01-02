import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import random
from pathlib import Path


class TextInfillingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, mask_ratio=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.mask_token_id = tokenizer.mask_token_id

    def __len__(self):
        return len(self.texts)
    
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        mask_pos = torch.zeros_like(input_ids, dtype=torch.bool)

        special = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ]

        for i in range(len(input_ids)):
            if input_ids[i].item() not in special:
                if random.random() < self.mask_ratio:
                    mask_pos[i] = True
                    input_ids[i] = self.mask_token_id

        return input_ids, labels, mask_pos
    
    def __getitem__(self, idx):
        text = self.texts[idx]

        enc = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        input_ids = enc['input_ids'].squeeze(0)
        attn_mask = enc['attention_mask'].squeeze(0)

        masked_ids, labels, mask_pos = self.mask_tokens(input_ids.clone())

        return {
            'input_ids': masked_ids,
            'attention_mask': attn_mask,
            'labels': labels,
            'mask_positions': mask_pos
        }


def get_dataloaders(data_dir='aclImdb', batch_size=16, max_length=128, 
                   train_split=0.8, max_samples=None):
    print(f"loading data from {data_dir}...")
    
    # initialize tokenizer once
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # load all texts from both train and test folders
    all_texts = []
    
    for split_dir in ['train', 'test']:
        data_path = Path(data_dir) / split_dir
        
        for folder in ['pos', 'neg']:
            folder_path = data_path / folder
            
            if not folder_path.exists():
                print(f"warning: {folder_path} not found")
                continue
            
            for txt_file in folder_path.glob('*.txt'):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        
                        # clean html
                        text = text.replace('<br />', ' ').replace('<br/>', ' ')
                        text = ' '.join(text.split())
                        
                        # filter by length
                        if 50 < len(text) < 5000:
                            all_texts.append(text)
                except Exception as e:
                    continue
                
                # stop if we hit max_samples
                if max_samples and len(all_texts) >= max_samples:
                    break
            
            if max_samples and len(all_texts) >= max_samples:
                break
        
        if max_samples and len(all_texts) >= max_samples:
            break
    
    print(f"loaded {len(all_texts)} total samples")
    
    # create full dataset
    full_dataset = TextInfillingDataset(
        texts=all_texts,
        tokenizer=tokenizer,
        max_length=max_length,
        mask_ratio=0.15
    )
    
    # split 80/20
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"train: {len(train_dataset)}, val: {len(val_dataset)}")
    
    # create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, tokenizer