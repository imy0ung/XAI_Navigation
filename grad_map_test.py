from vision_models.grad_eclip_model import GradEclipModel
import numpy as np

model = GradEclipModel()

image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

heatmap = model.get_image_features(image)

print(heatmap)