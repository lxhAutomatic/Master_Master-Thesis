import torch
from diffusers import DiffusionPipeline
device = "cpu"
pipeline = DiffusionPipeline.from_pretrained("DanishH/stable-diffusion-chest-xray-2")
pipe = pipeline.to(device)

prompt = "Chest X-ray image of Pneumothorax people."
#image_1 = pipe(prompt = prompt, num_inference_steps = 100, num_images_per_prompt = 5).images[0]
#image_1 = pipe(prompt = prompt, num_inference_steps = 100, num_images_per_prompt = 3).images[0]
#img_1 = image_1.convert('L')
#img_1.save("sd_test_1.png")

image_2 = pipe(prompt = prompt, num_inference_steps = 100, num_images_per_prompt = 16)
img_2_0 = image_2.images[0].convert('L')
img_2_1 = image_2.images[1].convert('L')
img_2_2 = image_2.images[2].convert('L')
img_2_3 = image_2.images[3].convert('L')
img_2_4 = image_2.images[4].convert('L')
img_2_5 = image_2.images[5].convert('L')
img_2_6 = image_2.images[6].convert('L')
img_2_7 = image_2.images[7].convert('L')
img_2_8 = image_2.images[8].convert('L')
img_2_9 = image_2.images[9].convert('L')
img_2_10 = image_2.images[10].convert('L')
img_2_11 = image_2.images[11].convert('L')
img_2_12 = image_2.images[12].convert('L')
img_2_13 = image_2.images[13].convert('L')
img_2_14 = image_2.images[14].convert('L')
img_2_15 = image_2.images[15].convert('L')

img_2_0.save("sd_test_0.png")
img_2_1.save("sd_test_1.png")
img_2_2.save("sd_test_2.png")
img_2_3.save("sd_test_3.png")
img_2_4.save("sd_test_4.png")
img_2_5.save("sd_test_5.png")
img_2_6.save("sd_test_6.png")
img_2_7.save("sd_test_7.png")
img_2_8.save("sd_test_8.png")
img_2_9.save("sd_test_9.png")
img_2_10.save("sd_test_10.png")
img_2_11.save("sd_test_11.png")
img_2_12.save("sd_test_12.png")
img_2_13.save("sd_test_13.png")
img_2_14.save("sd_test_14.png")
img_2_15.save("sd_test_15.png")