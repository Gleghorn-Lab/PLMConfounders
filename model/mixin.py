import torch
import os
from typing import Optional, List, Callable, Tuple
from transformers import PreTrainedTokenizerBase
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class ProteinDataset(Dataset):
    """Simple dataset for protein sequences."""
    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def build_collator(tokenizer) -> Callable[[list[str]], tuple[torch.Tensor, torch.Tensor]]:
    def _collate_fn(sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for batching sequences."""
        return tokenizer(sequences, return_tensors="pt", padding='longest')
    return _collate_fn


class PPIEmbeddingMixin:
    def _embed(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process embeddings through the PPI model.
        
        Args:
            embeddings: Pre-computed embeddings tensor of shape (batch_size, seq_len, embedding_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Processed embeddings of shape (batch_size, n_tokens, output_size)
        """
        raise NotImplementedError("Should be implemented by subclass")

    def _pad_embeddings_batch(self, embeddings_list: List[torch.Tensor], embed_dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of embeddings to the same length and create attention masks.
        
        Args:
            embeddings_list: List of embedding tensors with potentially different lengths
            embed_dtype: Target dtype for embeddings
            
        Returns:
            Tuple of (padded_embeddings, attention_mask)
        """
        if not embeddings_list:
            raise ValueError("embeddings_list cannot be empty")
            
        max_len = max(emb.shape[0] for emb in embeddings_list)
        batch_size = len(embeddings_list)
        embed_dim = embeddings_list[0].shape[-1]
        
        # Create padded tensor and attention mask
        padded_embeddings = torch.zeros(batch_size, max_len, embed_dim, dtype=embed_dtype)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float32)
        
        for i, emb in enumerate(embeddings_list):
            seq_len = emb.shape[0]
            padded_embeddings[i, :seq_len] = emb.to(embed_dtype)
            attention_mask[i, :seq_len] = 1.0
            
        return padded_embeddings, attention_mask

    def _process_embedding_batch(
        self, 
        batch_seqs: List[str], 
        input_embedding_dict: dict[str, torch.Tensor],
        embed_fn: Callable,
        embed_dtype: torch.dtype = torch.float32
    ) -> List[torch.Tensor]:
        """
        Process a batch of sequences through the embedding function.
        
        Args:
            batch_seqs: List of sequences in this batch
            input_embedding_dict: Dictionary mapping sequences to embeddings
            embed_fn: Function to process embeddings
            embed_dtype: Target dtype for embeddings
            
        Returns:
            List of processed embeddings for each sequence
        """
        # Extract embeddings for this batch
        batch_embeddings = [input_embedding_dict[seq] for seq in batch_seqs]
        
        # Pad embeddings and create attention mask
        padded_embeddings, attention_mask = self._pad_embeddings_batch(batch_embeddings, embed_dtype)
        
        # Move to device and process
        device = self.device
        padded_embeddings = padded_embeddings.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            processed_embeddings = embed_fn(padded_embeddings, attention_mask)
        
        return list(processed_embeddings)

    def _embed_from_dict(self, embedding_dict: dict, sequences: List[str], attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Helper method to extract embeddings from dictionary and process them.
        
        Args:
            embedding_dict: Dictionary mapping sequences to embeddings
            sequences: List of sequences to extract embeddings for
            attention_mask: Optional attention mask (will be auto-generated if None)
            
        Returns:
            Processed embeddings
        """
        # Extract embeddings from dictionary
        embeddings_list = [embedding_dict[seq] for seq in sequences]
        
        # Pad embeddings and create attention mask
        padded_embeddings, auto_attention_mask = self._pad_embeddings_batch(embeddings_list)
        
        # Use provided attention mask or auto-generated one
        if attention_mask is None:
            attention_mask = auto_attention_mask
            
        return self._embed(padded_embeddings, attention_mask)

    def embed_from_embeddings(
        self,
        input_embedding_dict: dict[str, torch.Tensor],
        sequences: Optional[List[str]] = None,
        batch_size: int = 2,
        embed_dtype: torch.dtype = torch.float32,
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'ppi_embeddings.db',
        save_path: str = 'ppi_embeddings.pth',
        embed_fn: Optional[Callable] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        """
        Process embeddings from another embedding dictionary (e.g., from PLMEmbeddingMixin).
        
        Args:
            input_embedding_dict: Dictionary mapping sequences to pre-computed embeddings
            sequences: List of sequences to process. If None, processes all sequences in input_embedding_dict
            batch_size: Batch size for processing
            embed_dtype: Output embedding dtype
            num_workers: Number of workers for data loading
            sql: Whether to store embeddings in SQLite database
            save: Whether to save embeddings
            sql_db_path: Path to SQLite database
            save_path: Path to save embeddings
            embed_fn: Optional custom embedding function (defaults to self._embed)
            
        Returns:
            Dictionary mapping sequences to processed embeddings, or None if sql=True
        """
        # Prepare sequences and validate inputs
        sequences = self._prepare_sequences(input_embedding_dict, sequences)
        if not sequences:
            return {} if not sql else None
            
        # Set default embedding function
        if embed_fn is None:
            embed_fn = self._embed
        
        # Determine which sequences need embedding
        to_embed = self._get_sequences_to_embed(sequences, sql, sql_db_path, save_path)
        
        if len(to_embed) > 0:
            if sql:
                self._process_and_store_sql(to_embed, input_embedding_dict, batch_size, embed_dtype, embed_fn, sql_db_path)
                return None
            else:
                embeddings_dict = self._load_existing_embeddings(save_path)
                self._process_and_store_dict(to_embed, input_embedding_dict, batch_size, embed_dtype, embed_fn, embeddings_dict)
                
                if save:
                    torch.save(embeddings_dict, save_path)
                    
                return embeddings_dict
        
        # Return empty dict or None if nothing to process
        return {} if not sql else None

    def _prepare_sequences(self, input_embedding_dict: dict[str, torch.Tensor], sequences: Optional[List[str]]) -> List[str]:
        """Prepare and validate sequences for processing."""
        if sequences is None:
            sequences = list(input_embedding_dict.keys())
        
        # Filter sequences that exist in input embedding dict
        available_sequences = [seq for seq in sequences if seq in input_embedding_dict]
        if len(available_sequences) != len(sequences):
            missing = len(sequences) - len(available_sequences)
            print(f"Warning: {missing} sequences not found in input_embedding_dict")
        
        if len(available_sequences) == 0:
            print("No sequences to process")
            return []
            
        # Sort by embedding length for efficient batching
        return sorted(available_sequences, key=lambda s: input_embedding_dict[s].shape[0], reverse=True)

    def _get_sequences_to_embed(self, sequences: List[str], sql: bool, sql_db_path: str, save_path: str) -> List[str]:
        """Determine which sequences need to be embedded."""
        if sql:
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
        else:
            if os.path.exists(save_path):
                existing_dict = torch.load(save_path, map_location='cpu', weights_only=True)
                to_embed = [seq for seq in sequences if seq not in existing_dict]
                print(f"Found {len(existing_dict)} already embedded sequences in {save_path}")
            else:
                to_embed = sequences
                
        print(f"Embedding {len(to_embed)} new sequences")
        return to_embed

    def _load_existing_embeddings(self, save_path: str) -> dict[str, torch.Tensor]:
        """Load existing embeddings if available."""
        if os.path.exists(save_path):
            return torch.load(save_path, map_location='cpu', weights_only=True)
        return {}

    def _process_and_store_sql(
        self, 
        to_embed: List[str], 
        input_embedding_dict: dict[str, torch.Tensor],
        batch_size: int,
        embed_dtype: torch.dtype,
        embed_fn: Callable,
        sql_db_path: str
    ):
        """Process embeddings and store in SQL database."""
        import sqlite3
        conn = sqlite3.connect(sql_db_path)
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
        
        for i in tqdm(range(0, len(to_embed), batch_size), desc='Processing embedded batches (SQL)'):
            batch_seqs = to_embed[i:i + batch_size]
            processed_embeddings = self._process_embedding_batch(batch_seqs, input_embedding_dict, embed_fn, embed_dtype)
            
            # Store in database
            for seq, emb in zip(batch_seqs, processed_embeddings):
                emb_reshaped = emb.reshape(-1, emb.shape[-1]).cpu().float()
                c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                        (seq, emb_reshaped.numpy().tobytes()))
            
            if (i // batch_size + 1) % 100 == 0:
                conn.commit()
        
        conn.commit()
        conn.close()

    def _process_and_store_dict(
        self,
        to_embed: List[str],
        input_embedding_dict: dict[str, torch.Tensor],
        batch_size: int,
        embed_dtype: torch.dtype,
        embed_fn: Callable,
        embeddings_dict: dict[str, torch.Tensor]
    ):
        """Process embeddings and store in dictionary."""
        for i in tqdm(range(0, len(to_embed), batch_size), desc='Processing embedded batches (PTH)'):
            batch_seqs = to_embed[i:i + batch_size]
            processed_embeddings = self._process_embedding_batch(batch_seqs, input_embedding_dict, embed_fn, embed_dtype)
            
            # Store processed embeddings
            for seq, emb in zip(batch_seqs, processed_embeddings):
                embeddings_dict[seq] = emb.cpu().to(embed_dtype)

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        import sqlite3
        import os
        sequences = []
        
        # If database file doesn't exist, return empty set
        if not os.path.exists(db_path):
            return set()
            
        try:
            with sqlite3.connect(db_path) as conn:
                c = conn.cursor()
                # Check if table exists first
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
                if c.fetchone() is None:
                    return set()
                
                c.execute("SELECT sequence FROM embeddings")
                while True:
                    row = c.fetchone()
                    if row is None:
                        break
                    sequences.append(row[0])
        except sqlite3.Error as e:
            print(f"SQLite error reading from {db_path}: {e}")
            return set()
            
        return set(sequences)

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 2,
        max_length: int = 2048,
        full_embeddings: bool = False,
        embed_dtype: torch.dtype = torch.float32,
        pooling_types: List[str] = ['mean'],
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
        embed_fn: Optional[Callable] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Embed a dataset of protein sequences.
        
        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing
            max_len: Maximum sequence length
            full_embeddings: Whether to return full residue-wise (True) embeddings or pooled (False)
            pooling_type: Type of pooling ('mean' or 'cls')
            num_workers: Number of workers for data loading, 0 for the main process
            sql: Whether to store embeddings in SQLite database - will be stored in float32
            sql_db_path: Path to SQLite database
            
        Returns:
            Dictionary mapping sequences to embeddings, or None if sql=True

        Note:
            - If sql=True, embeddings can only be stored in float32
            - sql is ideal if you need to stream a very large dataset for training in real-time
            - save=True is ideal if you can store the entire embedding dictionary in RAM
            - sql will be used if it is True and save is True or False
            - If your sql database or .pth file is already present, they will be scanned first for already embedded sequences
            - Sequences will be truncated to max_len and sorted by length in descending order for faster processing

        Example:
            >>> embedder = EmbeddingMixin()
            >>> embedding_dict = embedder.embed_dataset(
                sequences=[
                    'MALWMRLLPLLALLALWGPDPAAA', ... # list of protein sequences
                ],
                batch_size=2, # adjust for your GPU memory
                max_len=512, # adjust for your needs
                full_embeddings=False, # if True, no pooling is performed
                embed_dtype=torch.float32, # cast to what dtype you want
                pooling_type=['mean', 'cls'], # more than one pooling type will be concatenated together
                num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
                sql=False, # if True, embeddings will be stored in SQLite database
                sql_db_path='embeddings.db',
                save=True, # if True, embeddings will be saved as a .pth file
                save_path='embeddings.pth',
            )
            >>> # embedding_dict is a dictionary mapping sequences to their embeddings as tensors for .pth or numpy arrays for sql
        """
        sequences = list(set([seq[:max_length] for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        hidden_size = self.config.hidden_size
        collate_fn = build_collator(tokenizer)
        device = self.device
        if embed_fn is None:
            embed_fn = self._embed

        if sql:
            import sqlite3
            conn = sqlite3.connect(sql_db_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=2 if num_workers > 0 else None,
                    collate_fn=collate_fn,
                    shuffle=False,
                )
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        embeddings = embed_fn(input_ids, attention_mask).float()
                        ### In V2, we get a 32 x 128 back, no need to use trim_to or attention_mask
                        for seq, emb in zip(seqs, embeddings):
                            emb = emb.reshape(-1, hidden_size).float()
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = torch.load(save_path, map_location='cpu', weights_only=True)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            dataset = ProteinDataset(to_embed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    embeddings = embed_fn(input_ids, attention_mask).to(embed_dtype)
                    for seq, emb in zip(seqs, embeddings):
                        emb = emb.reshape(-1, hidden_size)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict
    

class PLMEmbeddingMixin:
    def _embed(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device

    def _read_sequences_from_db(self, db_path: str) -> set[str]:
        """Read sequences from SQLite database."""
        import sqlite3
        import os
        sequences = []
        
        # If database file doesn't exist, return empty set
        if not os.path.exists(db_path):
            return set()
            
        try:
            with sqlite3.connect(db_path) as conn:
                c = conn.cursor()
                # Check if table exists first
                c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
                if c.fetchone() is None:
                    return set()
                
                c.execute("SELECT sequence FROM embeddings")
                while True:
                    row = c.fetchone()
                    if row is None:
                        break
                    sequences.append(row[0])
        except sqlite3.Error as e:
            print(f"SQLite error reading from {db_path}: {e}")
            return set()
            
        return set(sequences)

    def embed_dataset(
        self,
        sequences: List[str],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 2,
        max_len: int = 512,
        truncate: bool = True,
        num_workers: int = 0,
        sql: bool = False,
        save: bool = True,
        sql_db_path: str = 'embeddings.db',
        save_path: str = 'embeddings.pth',
        embed_dim: Optional[int] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        
        sequences = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
        sequences = sorted(sequences, key=len, reverse=True)
        if embed_dim is None:
            embed_dim = self.config.hidden_size
        collate_fn = build_collator(tokenizer)
        device = self.device
        
        if sql:
            import sqlite3
            conn = sqlite3.connect(sql_db_path)
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS embeddings (sequence text PRIMARY KEY, embedding blob)')
            already_embedded = self._read_sequences_from_db(sql_db_path)
            to_embed = [seq for seq in sequences if seq not in already_embedded]
            print(f"Found {len(already_embedded)} already embedded sequences in {sql_db_path}")
            print(f"Embedding {len(to_embed)} new sequences")
            if len(to_embed) > 0:
                dataset = ProteinDataset(to_embed)
                dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                        seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                        embeddings = self._embed(input_ids, attention_mask).float() # sql requires float32
                        for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                            emb = emb[mask.bool()].reshape(-1, embed_dim)
                            c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", 
                                    (seq, emb.cpu().numpy().tobytes()))
                        
                        if (i + 1) % 100 == 0:
                            conn.commit()
            
                conn.commit()
            conn.close()
            return None

        embeddings_dict = {}
        if os.path.exists(save_path):
            embeddings_dict = torch.load(save_path, map_location='cpu', weights_only=True)
            to_embed = [seq for seq in sequences if seq not in embeddings_dict]
            print(f"Found {len(embeddings_dict)} already embedded sequences in {save_path}")
            print(f"Embedding {len(to_embed)} new sequences")
        else:
            to_embed = sequences
            print(f"Embedding {len(to_embed)} new sequences")

        if len(to_embed) > 0:
            dataset = ProteinDataset(to_embed)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding batches'):
                    seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                    input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                    embeddings = self._embed(input_ids, attention_mask)
                    for seq, emb, mask in zip(seqs, embeddings, attention_mask):
                        emb = emb[mask.bool()].reshape(-1, embed_dim)
                        embeddings_dict[seq] = emb.cpu()

        if save:
            torch.save(embeddings_dict, save_path)

        return embeddings_dict