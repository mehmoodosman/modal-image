import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

# from huggingface_hub import login
# login(token = os.getenv("HF_TOKEN"))

def download_model():
    from diffusers import StableDiffusionPipeline
    import torch

    model_id = "dreamlike-art/dreamlike-anime-1.0"

    StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

image = (modal.Image.debian_slim()
        .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests")
        .run_function(download_model))

app = modal.App("dreamlike-anime-demo", image=image)

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
        from diffusers import StableDiffusionPipeline
        import torch

        model_id = "dreamlike-art/dreamlike-anime-1.0"

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        self.pipe.to("cuda")
        self.MODAL_API_KEY = os.getenv("MODAL_API_KEY")

    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):

        modal_api_key = request.headers.get("X-API-KEY")
        if modal_api_key != self.MODAL_API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Generate image with the prompt
        negative_prompt = 'simple background, duplicate, retro style, low quality, lowest quality, 1980s, 1990s, 2000s, 2005 2006 2007 2008 2009 2010 2011 2012 2013, bad anatomy, bad proportions, extra digits, lowres, username, artist name, error, duplicate, watermark, signature, text, extra digit, fewer digits, worst quality, jpeg artifacts, blurry'
        image = self.pipe(prompt, negative_prompt=negative_prompt, height=832, width=704).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getvalue(), media_type="image/jpg")