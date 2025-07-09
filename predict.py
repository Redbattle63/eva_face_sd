import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import cog

MODEL_NAME = "stabilityai/stable-diffusion-1.5"

class Predictor(cog.Predictor):
    def setup(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            revision="fp16"
        ).to("cuda")

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    @cog.input("prompt", type=str, default="portrait of eva_visage_v1, ultra-realistic, close-up, electric blue strands framing the face, turquoise eyes")
    @cog.input("num_inference_steps", type=int, default=30)
    @cog.input("guidance_scale", type=float, default=7.5)
    def predict(self, prompt, num_inference_steps, guidance_scale):
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        output_path = "/tmp/output.png"
        image.save(output_path)
        return output_path
