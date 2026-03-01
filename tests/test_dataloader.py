"""
Tests for TextImageDataLoader.
CLIP (tokenizer + text encoder) is mocked — no GPU or model downloads required.
"""
import sys
import pytest
import torch
from pathlib import Path
from torchvision.transforms import RandomHorizontalFlip
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "utils" / "code"))

from conftest import make_clip_tokenizer_mock, make_clip_text_encoder_mock


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_dataloader(tmp_flower_dir, range_, training=True, max_length=77, image_size=(32, 32)):
    """Instantiate TextImageDataLoader with fully mocked CLIP."""
    tokenizer_mock = make_clip_tokenizer_mock()
    encoder_mock = make_clip_text_encoder_mock(max_length=max_length)

    with patch("dataloader.CLIPTokenizer") as mock_tok_cls, \
         patch("dataloader.CLIPTextModel") as mock_enc_cls:

        mock_tok_cls.from_pretrained.return_value = tokenizer_mock
        mock_enc_cls.from_pretrained.return_value = encoder_mock

        # Patch cuda operations so they work on CPU
        with patch("dataloader.torch.cuda.empty_cache"):
            from dataloader import TextImageDataLoader
            ds = TextImageDataLoader(
                datadir=tmp_flower_dir,
                range=range_,
                image_size=image_size,
                max_text_length=max_length,
                device="cpu",
                training=training,
            )
    return ds


# ── Slice / length tests ─────────────────────────────────────────────────────

def test_slice_start_zero(tmp_flower_dir):
    """range=(0, 3) should give exactly 3 items."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3))
    assert len(ds) == 3


def test_slice_offset(tmp_flower_dir):
    """range=(3, 5) should give 2 items (files at indices 3 and 4)."""
    ds = make_dataloader(tmp_flower_dir, range_=(3, 5))
    assert len(ds) == 2


def test_slice_full(tmp_flower_dir):
    """range=(0, 5) on a 5-image dir should give 5 items."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 5))
    assert len(ds) == 5


def test_slice_no_overlap(tmp_flower_dir):
    """Train and val slices built from the same dir must not share indices."""
    ds_train = make_dataloader(tmp_flower_dir, range_=(0, 3))
    ds_val = make_dataloader(tmp_flower_dir, range_=(3, 5))
    train_files = set(ds_train.datalist)
    val_files = set(ds_val.datalist)
    assert train_files.isdisjoint(val_files)


# ── Transform tests ──────────────────────────────────────────────────────────

def test_train_transform_has_flip(tmp_flower_dir):
    """training=True must include RandomHorizontalFlip."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3), training=True)
    transform_types = [type(t) for t in ds.image_transform.transforms]
    assert RandomHorizontalFlip in transform_types


def test_val_transform_no_flip(tmp_flower_dir):
    """training=False must NOT include RandomHorizontalFlip."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3), training=False)
    transform_types = [type(t) for t in ds.image_transform.transforms]
    assert RandomHorizontalFlip not in transform_types


# ── Null embedding tests ─────────────────────────────────────────────────────

def test_null_embedding_batch_shape(tmp_flower_dir):
    """get_null_embedding(B) must return shape (B, max_length, hidden_dim)."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3), max_length=77)
    null_emb = ds.get_null_embedding(4)
    assert null_emb.shape == (4, 77, 768)


def test_null_embedding_is_contiguous_after_expand(tmp_flower_dir):
    """Expanded null embedding moved to a device should be usable."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3))
    null_emb = ds.get_null_embedding(2)
    # .to("cpu") on an expand view is safe; result is a valid tensor
    moved = null_emb.to("cpu")
    assert moved.shape == (2, 77, 768)


# ── __getitem__ tests ────────────────────────────────────────────────────────

def test_getitem_returns_image_and_embedding(tmp_flower_dir):
    """__getitem__ must return (image_tensor, text_embedding)."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3), image_size=(32, 32))
    image, embedding = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(embedding, torch.Tensor)


def test_getitem_image_shape(tmp_flower_dir):
    """Image tensor must be (3, H, W) with values in [-1, 1]."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3), image_size=(32, 32))
    image, _ = ds[0]
    assert image.shape == (3, 32, 32)
    assert image.min() >= -1.0 - 1e-5
    assert image.max() <= 1.0 + 1e-5


def test_getitem_embedding_shape(tmp_flower_dir):
    """Text embedding must be (max_length, hidden_dim)."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 3), max_length=77)
    _, embedding = ds[0]
    assert embedding.shape == (77, 768)


def test_embeddings_cache_length_matches_datalist(tmp_flower_dir):
    """Pre-computed embeddings cache must have one entry per dataset item."""
    ds = make_dataloader(tmp_flower_dir, range_=(0, 4))
    assert len(ds.embeddings_cache) == len(ds.datalist)
