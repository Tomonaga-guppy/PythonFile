import torch
ckpt = torch.load(r"C:\Users\Tomson\StrokeProject\ViTPose\models\wholebody.pth", map_location="cpu")
sd = ckpt.get("state_dict", ckpt)
keys = list(sd.keys())
print("num keys:", len(keys))
print("contains experts?:", any("mlp.experts" in k for k in keys))
print("sample:", [k for k in keys if "mlp.experts" in k][:5])
