from dataclasses import dataclass
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import librosa
import torch
from typing import Dict, List, Optional, Union
from transformers import (
    Wav2Vec2Processor,
)

class TorontoDataset(Dataset):
  def __init__(
    self,
      df,
      processor,
      audio_dir,
      sr=16_000,
      features_column="input_features",
      labels_column="labels"
  ):
    self.df = df.reset_index(drop=True)
    self.processor = processor
    self.audio_dir = audio_dir
    self.sr = sr
    self.features_column=features_column
    self.labels_column=labels_column

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    row = self.df.iloc[i]
    path = self.audio_dir / row["speaker_id"] / row["filename"]
    audio, _ = librosa.load(path, sr=self.sr)

    input_features = self.processor(
      audio, sampling_rate=self.sr, return_tensors="pt"
    )[self.features_column][0]
    labels = self.processor.tokenizer(row["label"]).input_ids

    return {self.features_column: input_features, self.labels_column: labels}

@dataclass
class DataCollatorCTCWithPadding:
  """
  CODE FROM https://github.com/respeecher/ukrainian_asr/blob/main/hf_train.py

  Data collator that will dynamically pad the inputs received.
  Args:
      processor (:class:`~transformers.Wav2Vec2Processor`)
          The processor used for proccessing the data.
      padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
          Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
          among:
          * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
          * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
          * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
      max_length (:obj:`int`, `optional`):
          Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
      max_length_labels (:obj:`int`, `optional`):
          Maximum length of the ``labels`` returned list and optionally padding length (see above).
      pad_to_multiple_of (:obj:`int`, `optional`):
          If set will pad the sequence to a multiple of the provided value.
          This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
          7.5 (Volta).
  """

  processor: Wav2Vec2Processor
  padding: Union[bool, str] = True
  max_length: Optional[int] = None
  max_length_labels: Optional[int] = None
  pad_to_multiple_of: Optional[int] = None
  pad_to_multiple_of_labels: Optional[int] = None

  def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
    # split inputs and labels since they have to be of different lengths and need
    # different padding methods
    input_features = [{"input_values": feature["input_values"]} for feature in features]
    label_features = [{"input_ids": feature["labels"]} for feature in features]

    batch = self.processor.pad(
      input_features,
      padding=self.padding,
      max_length=self.max_length,
      pad_to_multiple_of=self.pad_to_multiple_of,
      return_tensors="pt",
    )

    labels_batch = self.processor.tokenizer.pad(
      label_features,
      padding=self.padding,
      max_length=self.max_length_labels,
      pad_to_multiple_of=self.pad_to_multiple_of_labels,
      return_tensors="pt",
    )

    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    batch["labels"] = labels

    return batch

def sanitize(df):
  df = df.copy()
  chars_to_ignore_regex = r'[\-,\?\.\!;:"«»]'
  ukrainian_chars = r"абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'’ "

  df["label"] = (
      df["label"].str.lower()
      .str.replace(chars_to_ignore_regex, "", regex=True)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
  )

  df = df[df["label"].str.len() > 0]

  disallowed_pattern = f"[^{ukrainian_chars}]"
  clean_mask = ~df["label"].str.contains(disallowed_pattern, regex=True)

  print(f"After filter: {clean_mask.sum()} / {len(df)} rows")

  return df[clean_mask]

class TimitDataset(Dataset):
  def __init__(
    self,
    df,
    processor,
    features_column="input_features",
    labels_column="labels"
  ):
    self.df = df.reset_index(drop=True)
    self.processor = processor
    self.sr = 16000
    self.features_column = features_column
    self.labels_column = labels_column

  def __len__(self):
    return len(self.df)

  def __getitem__(self, i):
    row = self.df.iloc[i]
    audio, _ = librosa.load(row["path"], sr=self.sr)

    input_features = self.processor(
      audio, sampling_rate=self.sr, return_tensors="pt"
    )[self.features_column][0]

    tok = self.processor.tokenizer
    phone_ids = tok.convert_tokens_to_ids(row["phonemes_39"])

    return {self.features_column: input_features, self.labels_column: phone_ids}

def build_timit_df(meta, base_dir, sr):
  audio = meta[meta["is_converted_audio"] == True]

  timit_df = []
  for _, row in audio.iterrows():
    wav = base_dir / row["path_from_data_dir"]
    phn = wav.with_name(row["filename"].replace(".WAV.wav", ".PHN"))
    phonemes = [line.split()[2] for line in phn.read_text().splitlines()]
    timit_df.append({
      "path": str(wav),
      "filename": row["filename"],
      "speaker_id": row["speaker_id"],
      "dialect_region": row["dialect_region"],
      "sentence_type": row["filename"][:2],
      "duration": librosa.get_duration(path=wav, sr=sr),
      "gender": row["speaker_id"][0],
      "phonemes": phonemes,
    })

  return pd.DataFrame(timit_df)

def map_phonemes(phonemes, phone_map):
  mapped = []
  unmapped = set()

  for p in phonemes:
    if p in phone_map:
      if phone_map[p] is not None:
        mapped.append(phone_map[p])
    else:
      unmapped.add(p)
      mapped.append("<unk>")

  if unmapped:
    print(f"Warning: phones not in PHONE_MAP: {unmapped}")

  return np.array(mapped, dtype=object)
