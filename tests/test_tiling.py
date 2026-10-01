from __future__ import annotations

import numpy as np

from jdll_denoise.tiling import batched_tiled_predict, tile_starts, tiled_predict


def test_tile_starts_cover_last_pixel() -> None:
    assert tile_starts(101, 32, 8)[-1] == 69


def test_identity_reconstructs_without_seams() -> None:
    source = np.random.default_rng(2).random((73, 91), dtype=np.float32)
    result = tiled_predict(source, lambda tile: tile, tile_size=32, overlap=12)
    np.testing.assert_allclose(result, source, atol=2e-7)


def test_batched_identity_reconstructs_and_reports_real_tiles() -> None:
    source = np.random.default_rng(4).random((73, 91), dtype=np.float32)
    result, metadata = batched_tiled_predict(
        source,
        lambda batch: batch,
        tile_size=32,
        overlap=12,
        batch_size=3,
    )
    np.testing.assert_allclose(result, source, atol=2e-7)
    assert metadata["inference_tile_count"] > 3
    assert metadata["inference_tile_batch_size"] == 3


def test_batched_prediction_halves_batch_after_simulated_oom() -> None:
    source = np.random.default_rng(5).random((64, 64), dtype=np.float32)

    def limited(batch: np.ndarray) -> np.ndarray:
        if len(batch) > 2:
            raise RuntimeError("out of memory in simulated accelerator")
        return batch

    result, metadata = batched_tiled_predict(
        source,
        limited,
        tile_size=32,
        overlap=8,
        batch_size=4,
    )
    np.testing.assert_allclose(result, source, atol=2e-7)
    assert metadata["inference_tile_batch_size"] == 2
    assert metadata["inference_tile_batch_reductions"] == [{"from": 4, "to": 2}]
