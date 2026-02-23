"""Monkey-patches for CrabNet to fix elem_prop propagation bugs.

CrabNet v2.0.8 has two bugs that cause the ``elem_prop`` parameter to be
silently ignored, defaulting every model to ``mat2vec`` embeddings:

Bug 1 — ``CrabNet.fit()`` instantiates ``SubCrab`` without forwarding
         ``self.elem_prop``.
Bug 2 — ``Encoder.__init__()`` instantiates ``Embedder`` without forwarding
         ``self.elem_prop``.

Additionally, many built-in element property CSVs cover fewer than 118
elements (e.g. jarvis has 82, oliynyk has 85), but CrabNet's EDM data
loader maps elements to a *fixed* 118-element list.  After fixing Bug 2
the Embedder may have too few rows — we pad with zeros so that all 118
elements are valid look-up indices.

Call :func:`patch_crabnet_elem_prop` once before any ``CrabNet.fit()``
invocation to fix all three issues.
"""

from __future__ import annotations

_PATCHED = False

# Number of entries the Embedder must have: 1 (padding) + 118 (elements H–Og)
_N_EMBEDDING_ROWS = 119


def patch_crabnet_elem_prop() -> None:
    """Apply monkey-patches so ``elem_prop`` propagates correctly.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _PATCHED
    if _PATCHED:
        return

    import torch
    from crabnet.crabnet_ import CrabNet
    from crabnet.kingcrab import SubCrab, Encoder, Embedder

    _data_type_torch = torch.float32

    # ------------------------------------------------------------------
    # Bug 2 fix + padding: Encoder.__init__ → Embedder must receive
    # elem_prop, and the resulting embedding is padded to 119 rows.
    # ------------------------------------------------------------------
    _original_encoder_init = Encoder.__init__

    def _patched_encoder_init(self, *args, **kwargs):
        elem_prop = kwargs.get("elem_prop", "mat2vec")
        _original_encoder_init(self, *args, **kwargs)
        # Re-create the Embedder with the correct elem_prop
        self.embed = Embedder(
            d_model=self.d_model,
            compute_device=self.compute_device,
            elem_prop=elem_prop,
        )
        # Pad embedding to cover all 118 elements if needed
        current_rows = self.embed.cbfv.weight.shape[0]
        if current_rows < _N_EMBEDDING_ROWS:
            feat_size = self.embed.cbfv.weight.shape[1]
            # Create pad on same device as existing weights to avoid device mismatch
            pad = torch.zeros(
                _N_EMBEDDING_ROWS - current_rows, feat_size, dtype=_data_type_torch,
                device=self.embed.cbfv.weight.device,
            )
            new_weight = torch.cat([self.embed.cbfv.weight.data, pad], dim=0)
            self.embed.cbfv = torch.nn.Embedding.from_pretrained(new_weight)
            if self.compute_device is not None:
                self.embed.cbfv = self.embed.cbfv.to(
                    self.compute_device, dtype=_data_type_torch,
                )

    Encoder.__init__ = _patched_encoder_init

    # ------------------------------------------------------------------
    # Bug 1 fix: CrabNet.fit() → SubCrab must receive elem_prop
    # ------------------------------------------------------------------
    _original_fit = CrabNet.fit

    def _patched_fit(self, *args, **kwargs):
        _orig_subcrab_init = SubCrab.__init__
        outer_elem_prop = self.elem_prop          # capture from CrabNet

        def _subcrab_init_with_prop(subcrab_self, *a, **kw):
            kw.setdefault("elem_prop", outer_elem_prop)
            _orig_subcrab_init(subcrab_self, *a, **kw)

        SubCrab.__init__ = _subcrab_init_with_prop
        try:
            return _original_fit(self, *args, **kwargs)
        finally:
            SubCrab.__init__ = _orig_subcrab_init

    CrabNet.fit = _patched_fit

    _PATCHED = True
    print("[patch] CrabNet elem_prop propagation fixed.")
