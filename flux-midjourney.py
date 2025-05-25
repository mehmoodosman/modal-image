import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

from huggingface_hub import login

login(token = os.getenv("HF_TOKEN"))

def download_model():
    from diffusers import DiffusionPipeline
    import torch

    base_model = "black-forest-labs/FLUX.1-dev"

    DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

image = (modal.Image.debian_slim()
        .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests", "sentencepiece", "peft")
        .run_function(download_model))

app = modal.App("flux-midjourney", image=image)

@app.cls(
    image=image,
    gpu="H100",     # GPU Selection
    container_idle_timeout=300,     # 5 minutes
    secrets=[modal.Secret.from_name("custom-secret")]
)
class Model:

    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import DiffusionPipeline
        import torch
        from peft import PeftModel

        base_model = "black-forest-labs/FLUX.1-dev"

        self.pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        lora_repo = "strangerzonehf/Flux-Midjourney-Mix2-LoRA"

        # self.pipe = PeftModel.from_pretrained(self.pipe, lora_repo)    

        self.pipe.load_lora_weights(lora_repo)
        
        self.pipe.to("cuda")
        self.MODAL_API_KEY = os.getenv("MODAL_API_KEY")

    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):

        modal_api_key = request.headers.get("X-API-KEY")
        if modal_api_key != self.MODAL_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Generate image with the prompt
        image = self.pipe(prompt).images[0] 

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getvalue(), media_type="image/jpg")