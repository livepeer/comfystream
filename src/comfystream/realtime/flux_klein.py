#!/usr/bin/env python3
# Realtime FLUX.2-klein-4B wrapper with a Krea-style feedback loop.
#
# SELF-CONTAINED: this is our own inference code (diffusers + Flux2KleinPipeline),
# NOT the Daydream Scope platform. The feedback-loop recipe is adapted from the
# reference plugin hthillman/scope-flux-klein — we borrow the *idea*, not the
# runtime. The runner (runner.py) calls process() once per incoming video frame.
#
# The trick (why Klein is usable in realtime):
#   frame 0      -> full inference (text_to_image / image_to_image), cached.
#   frame 1..N   -> refine_frame(): partial-denoise the PREVIOUS output instead
#                   of generating from scratch. In video mode the previous output
#                   is blended 50/50 with the incoming frame first, so the model
#                   refines something that already carries the prompt rather than
#                   denoising the raw camera feed. strength controls how many of
#                   the N steps actually run (0.5 -> ~2 of 4 steps).
#
# Flux2KleinPipeline has NO `strength` parameter (it conditions on the image via
# joint attention), so refine_frame() hand-rolls a truncated flow-matching
# schedule against diffusers internals. That ties us to a diffusers build that
# ships Flux2KleinPipeline (>= 0.37.0.dev0 / install from git).
from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("flux-klein")

DEFAULT_MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
DEFAULT_PROMPT = "a psychedelic landscape, vivid colors, intricate details"
DEFAULT_WIDTH = 384
DEFAULT_HEIGHT = 384
DEFAULT_STEPS = 4  # full-inference steps; refine uses a fraction of these
DEFAULT_GUIDANCE = 1.0  # Klein works best at 1.0 (CFG effectively off)
DEFAULT_FEEDBACK = 0.5  # 0 = regenerate fully each frame; 0.3 ~1 step, 0.5 ~2 of 4
DEFAULT_SEED = int(os.environ.get("FLUX_SEED", "-1"))  # -1 = fresh random noise every
# frame (organic shimmer); any fixed value reuses the SAME noise each frame, which
# visibly steadies the feedback loop (less flicker) and makes output reproducible.
DEFAULT_INPUT_BLEND = float(os.environ.get("FLUX_INPUT_BLEND", "0.5"))  # camera weight in
# the prev-output/camera blend (0..1). Higher = anchors to the real feed (less "fantasy"
# drift, weaker prompt); lower = prompt influence compounds harder across frames.
DEFAULT_BATCH = int(os.environ.get("FLUX_BATCH", "1"))  # frames refined per GPU pass.
# 1 (default) = today's behaviour exactly. >1 collects N frames and refines them all from
# the SAME previous output in one transformer/VAE pass: at 384x384 the GPU is far from
# saturated at batch 1, so B=2 costs ~1.3x the wall time of B=1 for 2x the frames. Costs
# latency (you must wait for N frames to arrive) and makes feedback stride-N — see README.
DEFAULT_MAX_TEXT_TOKENS = int(os.environ.get("FLUX_MAX_TEXT_TOKENS", "128"))  # prompt token
# budget. The DiT attends over text+image tokens jointly, so padding the text to the
# pipeline default of 512 makes the transformer carry ~512 dead tokens next to only 576
# image tokens (384x384). Short prompts fit in 128 easily; raise it if a long prompt
# gets truncated. The per-prompt log line in _refine shows the actual embed length.


def _snap(value: int, multiple: int = 16) -> int:
    """FLUX requires spatial dims to be multiples of 16."""
    return max(multiple, (value // multiple) * multiple)


# --- boundary conversions (numpy uint8 RGB <-> tensors) ---------------------


def _numpy_to_thwc(
    frame_rgb: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """HxWx3 uint8 RGB -> (1, H, W, 3) float [0, 1] on `device`.
    Uploads uint8 (4x smaller than fp32) and converts on the GPU. `dtype` defaults
    to fp32 but the feedback path passes the model dtype so the blend never has to
    round-trip through fp32."""
    t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).to(device)
    return t.to(dtype).div_(255.0).unsqueeze(0)


class FluxKleinModel:
    """Container-wide FLUX Klein pipeline. Loaded once at warm-up; process() is
    called per frame from the trickle loop. Resolution/model are fixed per
    container; only the prompt changes at runtime (via update_prompt)."""

    def __init__(self) -> None:
        self.pipe = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._width = DEFAULT_WIDTH
        self._height = DEFAULT_HEIGHT
        self._steps = DEFAULT_STEPS
        self._guidance = DEFAULT_GUIDANCE
        self._feedback = DEFAULT_FEEDBACK
        self._prompt = DEFAULT_PROMPT
        self._seed = DEFAULT_SEED
        self._input_blend = DEFAULT_INPUT_BLEND
        self._batch = DEFAULT_BATCH

        # Krea feedback loop: last emitted frame, kept as THWC [0,1] on device.
        self._prev: torch.Tensor | None = None

        # Per-frame caches (recomputing these every frame would dominate latency).
        self._cached_prompt_str: str | None = None
        self._cached_prompt_embeds: torch.Tensor | None = None
        self._cached_text_ids: torch.Tensor | None = None
        self._bn_mean: torch.Tensor | None = None
        self._bn_std: torch.Tensor | None = None
        self._bn_inv_std: torch.Tensor | None = None
        self._cached_latent_shape: tuple | None = None
        self._cached_latent_ids: torch.Tensor | None = None
        self._cached_timesteps_key: tuple | None = None
        self._cached_mu: float | None = None
        self._noise_key: tuple | None = None
        self._noise: torch.Tensor | None = None
        self._dtype: torch.dtype = torch.float16
        # Pinned staging buffer for the per-frame device->host copy (see _thwc_to_numpy).
        self._pinned: torch.Tensor | None = None
        # One-shot flag so the prompt-embed shape log fires on prompt change, not per frame.
        self._log_embed_shapes = False

    # --- lifecycle ----------------------------------------------------------
    def load(
        self,
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        feedback_strength: float = DEFAULT_FEEDBACK,
        seed: int = DEFAULT_SEED,
        input_blend: float = DEFAULT_INPUT_BLEND,
        batch: int = DEFAULT_BATCH,
        enable_cpu_offload: bool = False,
        compile_mode: "str | None" = None,
    ) -> None:
        self._width = _snap(width)
        self._height = _snap(height)
        self._steps = steps
        self._guidance = guidance
        self._feedback = feedback_strength
        self._prompt = prompt
        self._seed = seed
        self._input_blend = input_blend
        self._batch = max(1, int(batch))

        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as exc:
            raise ImportError(
                "Flux2KleinPipeline not found — needs a diffusers build with FLUX.2 "
                "support (>= 0.37.0.dev0). Install from git: "
                "pip install git+https://github.com/huggingface/diffusers.git"
            ) from exc

        # Shapes are static here (fixed resolution, fixed step count), so cudnn's
        # autotuner pays for itself on the first frames instead of thrashing; TF32
        # matmuls apply to whatever fp32 work remains (VAE bits, compile guards).
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        log.info("loading %s (%dx%d, %d steps)", model, self._width, self._height, self._steps)
        self.pipe = Flux2KleinPipeline.from_pretrained(model, torch_dtype=torch.float16)
        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self._device)
        # Cache once: after torch.compile this would resolve through
        # OptimizedModule.__getattr__ on every frame.
        self._dtype = self.pipe.transformer.dtype

        if compile_mode:
            # torch.compile the transformer (the per-step cost). Compilation is
            # triggered by the dummy passes below so it lands during warm-up
            # (health still 503) instead of stalling the first live session.
            # Two passes: full-inference and refine call the transformer with
            # different shapes, and each shape compiles separately.
            # compile_mode is passed to torch.compile verbatim: "default" (fast
            # compile), "max-autotune" (much longer warm-up, fastest kernels), or
            # "reduce-overhead" (CUDA graphs — shapes here are static so it applies).
            log.info("compiling transformer + VAE (mode=%s; adds minutes to warm-up)", compile_mode)
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode=compile_mode)
            # The VAE runs a full encode AND decode every refine frame — with only
            # ~2 transformer steps at feedback 0.5, it's a large share of frame time
            # and was the biggest remaining uncompiled block. Compile the encoder /
            # decoder submodules, not pipe.vae: encode()/decode() are wrappers that
            # do tiling/slicing and would graph-break.
            self.pipe.vae.to(memory_format=torch.channels_last)
            self.pipe.vae.encoder = torch.compile(self.pipe.vae.encoder, mode=compile_mode)
            self.pipe.vae.decoder = torch.compile(self.pipe.vae.decoder, mode=compile_mode)
            dummy = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            self.process(dummy)  # compiles the full-inference path (first frame)
            self.process(dummy)  # compiles the refine path (every frame after)
            self.process(dummy)  # second refine pass: settles CUDA-graph capture
            if self._batch > 1:
                # A batched refine is a *different* shape to the compiler, so it would
                # otherwise trigger a fresh (multi-minute under max-autotune) compile on
                # the first live batch — i.e. after /health already said ready. Compile
                # it here. Note both shapes stay live: a stream tail can hand the worker
                # a partial batch of 1, which uses the B=1 graph compiled above.
                self.process_batch([dummy] * self._batch)
                self.process_batch([dummy] * self._batch)  # settles CUDA-graph capture
            self._prev = None  # don't leak warm-up state into the first session
            log.info("compile warm-up done")

        log.info("FLUX Klein ready")

    def reset(self) -> None:
        """Drop the feedback state. MUST be called between sessions — otherwise the
        next session's first frame refines the previous client's last output instead
        of doing a fresh full inference."""
        self._prev = None

    def update_prompt(self, prompt: str) -> None:
        self._prompt = prompt  # prompt-embed cache invalidates itself on change

    def update_seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._noise_key = None  # force regeneration for the new seed

    @property
    def seed(self) -> int:
        return self._seed

    def update_input_blend(self, input_blend: float) -> None:
        self._input_blend = min(1.0, max(0.0, float(input_blend)))

    @property
    def input_blend(self) -> float:
        return self._input_blend

    @property
    def batch(self) -> int:
        return self._batch

    def _make_generator(self) -> "torch.Generator | None":
        """Fresh generator with the SAME seed every call: a fixed seed therefore
        reuses identical noise each frame (steady, reproducible), while -1 means
        fully random noise per frame (organic shimmer)."""
        if self._seed < 0:
            return None
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._seed)
        return gen

    # --- per-frame entry point ---------------------------------------------
    def process(self, frame_rgb: np.ndarray | None) -> np.ndarray:
        """Run one feedback-loop step. frame_rgb is HxWx3 uint8 RGB (the incoming
        video frame) or None for pure text mode. Returns HxWx3 uint8 RGB."""
        if self.pipe is None:
            raise RuntimeError("pipeline not loaded — call load() first")

        prompt = self._prompt if self._prompt.strip() else "high quality image"

        if frame_rgb is not None:
            if self._prev is None or self._feedback <= 0:
                out = self._image_to_image(prompt, frame_rgb)  # first frame: full
            else:
                blended = self._blend_prev_with_frame(frame_rgb)  # steer toward the feed
                out = self._refine(prompt, blended, self._feedback)
        else:
            if self._prev is None or self._feedback <= 0:
                out = self._text_to_image(prompt)  # first frame: full
            else:
                out = self._refine(prompt, self._prev, self._feedback)

        self._prev = out.detach()
        return self._thwc_to_numpy(out)

    def process_batch(self, frames_rgb: "list[np.ndarray]") -> "list[np.ndarray]":
        """Refine N frames in ONE GPU pass and return N HxWx3 uint8 RGB frames, in
        input order. Every frame in the batch is blended against the SAME self._prev,
        so feedback advances once per batch (stride-N) rather than once per frame —
        that's the behavioural price of the throughput win. The newest output seeds
        the next batch. N == 1 is exactly process()."""
        if not frames_rgb:
            return []
        if self.pipe is None:
            raise RuntimeError("pipeline not loaded — call load() first")
        if len(frames_rgb) == 1:
            return [self.process(frames_rgb[0])]

        prompt = self._prompt if self._prompt.strip() else "high quality image"

        # Cold start: the full-inference path takes a PIL image and has no batch form,
        # so the first frame runs alone at B=1 to seed _prev; the rest batch normally.
        head: list[np.ndarray] = []
        if self._prev is None or self._feedback <= 0:
            head.append(self.process(frames_rgb[0]))
            frames_rgb = frames_rgb[1:]
            if not frames_rgb:
                return head
            if self._feedback <= 0:
                # Feedback disabled: every frame is a full inference, one at a time.
                return head + [self.process(f) for f in frames_rgb]

        blended = torch.cat([self._blend_prev_with_frame(f) for f in frames_rgb], dim=0)
        out = self._refine(prompt, blended, self._feedback)
        self._prev = out[-1:].detach()  # newest output seeds the next batch
        return head + [self._thwc_to_numpy(out[i : i + 1]) for i in range(out.shape[0])]

    def _thwc_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """(1, H, W, 3) float [0, 1] (any device) -> HxWx3 uint8 RGB. Takes ONE frame:
        batched callers slice (out[i:i+1]) and call this per element, which keeps the
        pinned buffer below a single fixed shape.
        Converts to uint8 on the GPU so only 1/4 of the bytes cross PCIe, then copies
        through a reusable pinned host buffer — a plain .cpu() on a pageable
        destination makes the driver stage through its own bounce buffer and blocks
        the whole copy on the critical path."""
        t = tensor[0].clamp(0, 1).mul(255.0).round().to(torch.uint8)
        if t.device.type != "cuda":  # CPU-only dev boxes: nothing to pin
            return t.numpy()
        if self._pinned is None or self._pinned.shape != t.shape:
            self._pinned = torch.empty(t.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
        self._pinned.copy_(t, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        # .copy() is load-bearing: the caller hands this array to PyAV
        # (VideoFrame.from_ndarray), which keeps a reference — returning a view of
        # the reusable buffer would let the next frame overwrite a frame still in
        # flight. The copy is a ~440 KB memcpy at 384x384, far cheaper than the
        # pageable transfer it replaces.
        return self._pinned.numpy().copy()

    def _blend_prev_with_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """Blend the previous output with the new frame at engine resolution
        (THWC [0,1]). Refining this (rather than the raw feed) keeps prompt
        influence alive; input_blend is the camera's weight — raise it to anchor
        the stream to reality, lower it to let the prompt take over."""
        # Match _prev's dtype (model dtype, fp16): the whole feedback path stays in
        # fp16 so nothing pays a full-frame fp32 pass. interpolate() is fine on fp16 CUDA.
        frame = _numpy_to_thwc(frame_rgb, self._prev.device, self._prev.dtype)
        # Resize the incoming frame to engine resolution so we blend against prev.
        if frame.shape[1] != self._height or frame.shape[2] != self._width:
            frame = torch.nn.functional.interpolate(
                frame.permute(0, 3, 1, 2),
                size=(self._height, self._width),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
        w = self._input_blend
        return (1.0 - w) * self._prev + w * frame

    # --- full inference (first frame / feedback disabled) -------------------
    @torch.no_grad()
    def _text_to_image(self, prompt: str) -> torch.Tensor:
        result = self.pipe(
            prompt=prompt,
            width=self._width,
            height=self._height,
            guidance_scale=self._guidance,
            num_inference_steps=self._steps,
            generator=self._make_generator(),
            output_type="pt",  # tensor on-device; skip the PIL round-trip
        )
        # images: (B, C, H, W) float in [0, 1] -> THWC on device. Left in model dtype:
        # _prev is fp16 throughout so the feedback path never upcasts.
        return result.images[0].permute(1, 2, 0).unsqueeze(0)

    @torch.no_grad()
    def _image_to_image(self, prompt: str, frame_rgb: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(frame_rgb, mode="RGB").resize(
            (self._width, self._height), Image.LANCZOS
        )
        result = self.pipe(
            prompt=prompt,
            image=image,
            width=self._width,
            height=self._height,
            guidance_scale=self._guidance,
            num_inference_steps=self._steps,
            generator=self._make_generator(),
            output_type="pt",  # tensor on-device; skip the PIL round-trip
        )
        return result.images[0].permute(1, 2, 0).unsqueeze(0)

    # --- feedback refine (the Krea trick) -----------------------------------
    def _get_bn_stats(self, device, dtype):
        """VAE batch-norm stats — constant after load. Returns (mean, std, inv_std):
        normalize multiplies by inv_std, denormalize multiplies by std, so neither
        hot path divides."""
        if self._bn_mean is None:
            self._bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
            self._bn_std = torch.sqrt(
                self.pipe.vae.bn.running_var.view(1, -1, 1, 1).to(device, dtype)
                + self.pipe.vae.config.batch_norm_eps
            )
            self._bn_inv_std = 1.0 / self._bn_std
        return self._bn_mean, self._bn_std, self._bn_inv_std

    def _get_latent_ids(self, latent_shape: tuple, device):
        # Keyed on the spatial dims only, so a future batch dim doesn't thrash it.
        key = tuple(latent_shape[1:])
        if self._cached_latent_shape != key:
            dummy = torch.empty(latent_shape)
            self._cached_latent_ids = self.pipe._prepare_latent_ids(dummy).to(device)
            self._cached_latent_shape = key
        return self._cached_latent_ids

    def _get_noise(self, shape: tuple, device, dtype) -> torch.Tensor:
        """Sampling noise for the refine step.

        Fixed seed: the same seed re-generates bit-identical noise every frame (that's
        what makes output steady), so generate ONCE and reuse — the previous code paid
        CPU RNG plus a full host->device copy per frame to recompute a constant.
        seed -1: fresh noise per frame, generated directly on the device.

        Batched (shape[0] > 1): the fixed-seed noise is generated at B=1 and expanded, so
        every element of the batch gets the SAME noise. Distinct per-element noise would
        silently break the "fixed seed = steady output" property — the batch elements
        would shimmer against each other even though the seed never changed. For seed -1
        per-element random noise is the correct behaviour (that path is the shimmer)."""
        from diffusers.utils.torch_utils import randn_tensor

        if self._seed < 0:
            return randn_tensor(shape, device=device, dtype=dtype)
        base_shape = (1,) + tuple(shape[1:])
        key = (self._seed, base_shape)
        if self._noise_key != key:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._seed)
            self._noise = randn_tensor(base_shape, generator=gen, device=device, dtype=dtype)
            self._noise_key = key
        # expand() is a free view; it's only ever read (the interpolation below is
        # non-inplace), so the broadcast stride is safe.
        return self._noise.expand(shape) if shape[0] != 1 else self._noise

    @torch.no_grad()
    def _refine(self, prompt: str, prev_thwc: torch.Tensor, strength: float) -> torch.Tensor:
        """Partial-denoise prev_thwc (THWC [0,1]) toward `prompt`. Adapted from
        scope-flux-klein's refine_frame: VAE-encode -> add noise at a partial
        timestep -> denoise only the remaining steps -> decode."""
        device = self.pipe._execution_device
        dtype = self._dtype
        t0 = time.perf_counter()

        # THWC [0,1] -> BCHW [-1,1] for the VAE, fused in model dtype (one pass, no
        # fp32 intermediate). Non-inplace on purpose: prev_thwc IS self._prev, which
        # the next frame's blend still reads.
        image_tensor = prev_thwc.to(device).permute(0, 3, 1, 2).to(dtype).mul(2.0).sub(1.0)

        # 1. Prompt embeddings (re-encode only when the prompt changes).
        if prompt != self._cached_prompt_str:
            self._cached_prompt_embeds, self._cached_text_ids = self.pipe.encode_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=DEFAULT_MAX_TEXT_TOKENS,
            )
            self._cached_prompt_str = prompt
            self._log_embed_shapes = True  # logged below, where image_seq_len is known
        prompt_embeds, text_ids = self._cached_prompt_embeds, self._cached_text_ids
        # Batched refine: the embeds are cached at B=1, so broadcast them across the
        # batch (a view, free — all elements share one prompt anyway). text_ids/img_ids
        # are NOT expanded: in the FLUX family they are unbatched (seq, 3) position ids
        # shared by the whole batch — img_ids comes from _get_latent_ids, whose cache key
        # is deliberately spatial-only for exactly that reason.
        batch_size = prev_thwc.shape[0]
        if batch_size > 1:
            prompt_embeds = prompt_embeds.expand(batch_size, *prompt_embeds.shape[1:])

        # 2. Encode -> patchify -> BN normalize -> pack.
        image_latents = self.pipe.vae.encode(image_tensor).latent_dist.mode()
        image_latents = self.pipe._patchify_latents(image_latents)
        bn_mean, bn_std, bn_inv_std = self._get_bn_stats(device, dtype)
        image_latents = (image_latents - bn_mean) * bn_inv_std
        latent_ids = self._get_latent_ids(image_latents.shape, device)
        clean_latents = self.pipe._pack_latents(image_latents)

        # 3. Truncated schedule: run only the last `strength * steps` steps.
        # set_timesteps MUST run every frame: besides rebuilding the (tiny) sigma
        # array it resets the scheduler's internal _step_index, which otherwise
        # keeps advancing across frames and indexes past the end of sigmas.
        # Only `mu` is cached — it depends solely on (seq_len, steps), both fixed.
        image_seq_len = clean_latents.shape[1]
        if self._log_embed_shapes:
            # The DiT attends over text+image jointly, so text padding is real work.
            # If the embed length here is much larger than the prompt, lower
            # FLUX_MAX_TEXT_TOKENS.
            log.info(
                "prompt embeds=%s text_ids=%s image_tokens=%d",
                tuple(self._cached_prompt_embeds.shape),
                tuple(self._cached_text_ids.shape),
                image_seq_len,
            )
            self._log_embed_shapes = False
        ts_key = (image_seq_len, self._steps)
        if self._cached_timesteps_key != ts_key:
            # Imported lazily so load()'s friendly ImportError still fires first.
            from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

            self._cached_mu = compute_empirical_mu(image_seq_len, self._steps)
            self._cached_timesteps_key = ts_key
        self.pipe.scheduler.set_timesteps(self._steps, device=device, mu=self._cached_mu)
        all_timesteps = self.pipe.scheduler.timesteps

        init_timestep = min(int(self._steps * strength), self._steps)
        t_start = max(self._steps - init_timestep, 0)
        timesteps = all_timesteps[t_start:]
        # Load-bearing: without it the scheduler falls back to a per-step
        # (timesteps == t).nonzero().item() lookup, i.e. a GPU->CPU sync every step.
        self.pipe.scheduler.set_begin_index(t_start)

        if len(timesteps) == 0:  # strength ~0: nothing to denoise
            return prev_thwc

        # 4. Add noise at the starting sigma (flow-matching interpolation).
        start_sigma = timesteps[0].float() / self.pipe.scheduler.config.num_train_timesteps
        start_sigma = start_sigma.view(-1, 1, 1).to(device=device, dtype=dtype)
        noise = self._get_noise(clean_latents.shape, device, dtype)
        latents = start_sigma * noise + (1.0 - start_sigma) * clean_latents

        # 5. Denoise the remaining steps (transformer called directly; CFG off at guidance=1).
        # Iterate on CPU: indexing a CUDA tensor in a Python loop materializes a 0-dim
        # GPU tensor (and a launch) per step. The scheduler only uses `t` for its
        # sigma bookkeeping — and with set_begin_index above it doesn't even look `t`
        # up — so a CPU scalar is fine and costs no sync.
        timesteps_cpu = timesteps.to("cpu")
        for t in timesteps_cpu:
            # t is a CPU scalar now, so move the expanded timestep to the device here.
            timestep = t.expand(latents.shape[0]).to(latents.device, latents.dtype)
            noise_pred = self.pipe.transformer(
                hidden_states=latents.to(dtype),
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :image_seq_len]
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 6. Unpack -> BN denormalize -> unpatchify -> VAE decode.
        latents = self.pipe._unpack_latents_with_ids(latents, latent_ids)
        latents = latents * bn_std + bn_mean  # inverse of the normalize above
        latents = self.pipe._unpatchify_latents(latents)
        decoded = self.pipe.vae.decode(latents.to(dtype), return_dict=False)[0]

        decoded = (decoded.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        # Stays in model dtype (fp16): _prev, the blend and the VAE input are all fp16,
        # so upcasting here only bought three extra full-frame passes per frame.
        out = decoded.permute(0, 2, 3, 1)  # BCHW -> (B,H,W,C)

        if log.isEnabledFor(logging.DEBUG):  # args eval eagerly; don't pay per frame
            dt = time.perf_counter() - t0
            log.debug(
                "refine: %d steps in %.3fs (%.1f fps potential)", len(timesteps), dt, 1.0 / dt
            )
        return out
