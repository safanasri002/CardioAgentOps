from pathlib import Path
import numpy as np
import torch
from audio_engine.train_deep import CNNCustom, TransformerCNN, make_resnet

NAME = "TransformerCNN"  # change selon le meilleur Recall
MODELS = Path("audio_engine/models")
X = np.load("data/processed/X_mel_2d.npy")
_, H, W = X.shape

builders = {"CNNCustom": CNNCustom, "ResNet18": make_resnet, "TransformerCNN": TransformerCNN}
model = builders[NAME]()
model.load_state_dict(torch.load(MODELS / f"{NAME}.pt", map_location="cpu"))
model.eval()

dummy = torch.zeros(1, 1, H, W)
out = MODELS / "cardio_model.onnx"
torch.onnx.export(
    model, dummy, out,
    input_names=["mel"], output_names=["logits"],
    dynamic_axes={"mel": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print("exporté", out)