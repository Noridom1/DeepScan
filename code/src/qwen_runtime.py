import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import io
import base64
import asyncio
from typing import Optional, Union, List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def _to_dtype(s: str) -> Union[str, torch.dtype]:
    s = (s or "auto").lower()
    if s in ("auto", ""):
        return "auto"
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


ImageInputType = Union[str, Image.Image]
MultiImageInputType = Union[ImageInputType, Sequence[ImageInputType]]


class QwenVLRuntime:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        attn_impl: Optional[str] = None,
        dtype: str = "auto",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_concurrency: int = 1,
        system_prompt: str = (
            "You are an advanced image understanding assistant. "
            "You will be given an image and a question about it."
        ),
    ):
        dtype_arg = _to_dtype(dtype)

        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "dtype": dtype_arg,
        }
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            **model_kwargs,
        )

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            **processor_kwargs,
        )

        self.system_prompt = system_prompt
        self.sem = asyncio.Semaphore(max_concurrency)

    async def generate(
        self,
        prompt: str,
        image_input: MultiImageInputType,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
    ) -> str:
        async with self.sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._blocking_generate,
                prompt,
                image_input,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                presence_penalty,
            )

    def _to_pil_image(self, image_input: ImageInputType) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if not isinstance(image_input, str):
            raise TypeError(
                f"Unsupported image_input type: {type(image_input)}. "
                "Expected str(base64/path/data-url) or PIL.Image."
            )

        s = image_input.strip()

        if os.path.exists(s):
            return Image.open(s).convert("RGB")

        if s.startswith("data:"):
            if "," not in s:
                raise ValueError("Invalid data URL image input: missing comma separator.")
            s = s.split(",", 1)[1].strip()

        s = "".join(s.split())
        pad_len = (-len(s)) % 4
        if pad_len:
            s += "=" * pad_len

        try:
            image_bytes = base64.b64decode(s, validate=False)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {e}") from e

        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Decoded bytes are not a valid image: {e}") from e

    def _to_pil_images(self, image_input: MultiImageInputType) -> List[Image.Image]:
        if isinstance(image_input, (list, tuple)):
            if len(image_input) == 0:
                raise ValueError("image_input list is empty.")
            return [self._to_pil_image(x) for x in image_input]

        return [self._to_pil_image(image_input)]

    def _blocking_generate(
        self,
        prompt: str,
        image_input: MultiImageInputType,
        max_tokens: int,
        temperature: float,
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        presence_penalty: Optional[float],
    ) -> str:
        pil_images = self._to_pil_images(image_input)

        full_prompt = prompt
        if self.system_prompt:
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

        content = []
        for pil_image in pil_images:
            content.append(
                {
                    "type": "image",
                    "image": pil_image,
                }
            )
        content.append(
            {
                "type": "text",
                "text": full_prompt,
            }
        )

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": temperature > 0,
        }

        if temperature > 0:
            gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)
            if top_k is not None:
                gen_kwargs["top_k"] = int(top_k)

        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        if presence_penalty is not None:
            gen_kwargs["presence_penalty"] = float(presence_penalty)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0] if output_text else ""


# ----------- lazy loading -----------
_singleton: Optional[QwenVLRuntime] = None
_singleton_lock: Optional[asyncio.Lock] = None


async def get_qwen_runtime(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    attn_impl: Optional[str] = None,
    dtype: str = "auto",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_concurrency: int = 1,
    system_prompt: str = (
        "You are an advanced image understanding assistant. "
        "You will be given an image and a question about it."
    ),
) -> QwenVLRuntime:
    global _singleton, _singleton_lock

    if _singleton is None:
        if _singleton_lock is None:
            _singleton_lock = asyncio.Lock()

        async with _singleton_lock:
            if _singleton is None:
                _singleton = QwenVLRuntime(
                    model_name=model_name,
                    attn_impl=attn_impl,
                    dtype=dtype,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    max_concurrency=max_concurrency,
                    system_prompt=system_prompt,
                )
    return _singleton