# src/etude/data/vocab.py

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Union

# --- Special Tokens ---
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


@dataclass
class Event:
    """
    A dataclass representing a single musical event.

    This is the fundamental unit of representation that is converted
    into a token and then an integer ID by the Vocab.

    Attributes:
        type_ (str): The category of the event (e.g., 'Note', 'Pos', 'Duration').
        value (Union[str, int]): The value of the event (e.g., 60 for MIDI pitch, 4 for a 16th note duration).
    """
    type_: str
    value: Union[str, int]

    def __str__(self) -> str:
        """Returns the string representation of the event, used as the token."""
        return f"{self.type_}_{self.value}"

    def __repr__(self) -> str:
        return f"Event(type={self.type_}, value={self.value})"


class Vocab:
    """
    Manages the mapping between string tokens (or Events) and integer IDs.

    This class handles vocabulary creation, saving, loading, and the encoding/decoding
    of individual tokens and sequences.
    """
    def __init__(self, special_tokens: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]):
        """
        Initializes the vocabulary.
        
        Args:
            special_tokens (List[str]): A list of special tokens to add first.
                                         These tokens will have fixed, low-integer IDs.
        """
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.special_tokens = special_tokens

        for token in self.special_tokens:
            self._add_token(token)

        print(f"Initialized Vocab with special tokens: {self.special_tokens}")

    def _add_token(self, token: str) -> int:
        """Adds a token to the vocabulary if it doesn't already exist."""
        if token not in self.token_to_id:
            token_id = len(self.id_to_token)
            self.token_to_id[token] = token_id
            self.id_to_token.append(token)
            return token_id
        
        return self.token_to_id[token]

    def build_from_events(self, event_sequences: List[List[Event]]):
        """
        Builds or updates the vocabulary from a list of event sequences.

        Args:
            event_sequences (List[List[Event]]): A list of event sequences to build the vocabulary from.
        """
        print("Building vocabulary from event sequences...")

        for seq in event_sequences:
            for event in seq:
                self._add_token(str(event))

        print(f"Vocabulary built. Total unique tokens: {len(self)}")

    def encode(self, token: Union[str, Event]) -> int:
        """
        Converts a token (string or Event) to its integer ID.
        If the token is not found, it falls back to the UNK_TOKEN's ID.
        If UNK_TOKEN is not defined, it raises an error.
        """
        token_str = str(token)
        unk_token_id = self.token_to_id.get(UNK_TOKEN)

        token_id = self.token_to_id.get(token_str, unk_token_id)

        if token_id is None:
            # This case happens if a token is not in the vocab AND UNK_TOKEN was not defined.
            raise ValueError(
                f"Token '{token_str}' is not in the vocabulary, and no '{UNK_TOKEN}' is defined "
                f"in the vocabulary to use as a fallback. Please ensure '{UNK_TOKEN}' is in the "
                "special_tokens list when creating the vocabulary."
            )
        
        return token_id

    def decode(self, token_id: int) -> str:
        """Converts an integer ID back to its string token."""
        if 0 <= token_id < len(self.id_to_token):
            return self.id_to_token[token_id]
        
        raise ValueError(f"Invalid token ID: {token_id}")

    def decode_to_event(self, token_id: int) -> Event:
        """Converts an integer ID back to an Event object."""
        token_str = self.decode(token_id)
        if token_str in self.special_tokens:
            return Event(type_=token_str, value='')

        try:
            type_, value_str = token_str.split('_', 1)
            # Attempt to convert value to int for known numeric event types
            value = int(value_str) if type_ in {"Note", "Pos", "TimeSig", "Duration", "Grace"} else value_str
        except (ValueError, IndexError):
            # Handle cases with no '_' or non-integer values
            type_, value = token_str, ''
        
        return Event(type_=type_, value=value)

    def encode_sequence(self, sequence: List[Union[str, Event]]) -> List[int]:
        """Converts a sequence of tokens/Events to a sequence of integer IDs."""
        return [self.encode(token) for token in sequence]

    def decode_sequence(self, id_sequence: List[int]) -> List[str]:
        """Converts a sequence of integer IDs back to a sequence of string tokens."""
        return [self.decode(token_id) for token_id in id_sequence if token_id != self.get_pad_id()]
    
    def decode_sequence_to_events(self, id_sequence: List[int]) -> List[Event]:
        """Converts a sequence of integer IDs back to a sequence of Event objects."""
        return [self.decode_to_event(token_id) for token_id in id_sequence if token_id != self.get_pad_id()]

    def save(self, filepath: Union[str, Path]):
        """Saves the vocabulary mapping to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Vocab':
        """Loads the vocabulary from a JSON file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        instance = cls(special_tokens=vocab_data.get('special_tokens', [PAD_TOKEN]))
        instance.token_to_id = vocab_data['token_to_id']
        # Reconstruct id_to_token from the loaded mapping
        instance.id_to_token = [""] * len(instance.token_to_id)
        for token, token_id in instance.token_to_id.items():
             instance.id_to_token[token_id] = token

        print(f"Vocabulary loaded from {filepath}. Size: {len(instance)}")
        return instance

    def __len__(self) -> int:
        return len(self.id_to_token)

    def get_pad_id(self) -> int:
        return self.token_to_id.get(PAD_TOKEN, -1)

    def get_bar_bos_id(self) -> int:
        return self.token_to_id.get('Bar_BOS', -1)

    def get_bar_eos_id(self) -> int:
        return self.token_to_id.get('Bar_EOS', -1)