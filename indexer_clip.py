from transformers import CLIPProcessor, CLIPModel #type: ignore
import torch
from PIL import Image
from shore_db import ShoreDB # type: ignore

class CLIPImageIndexer:
  def __init__(self, model_name="openai/clip-vit-base-patch16"):
    print(f"\nLoading Clip model {model_name}...")
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = CLIPModel.from_pretrained(model_name).to(self.device) #type: ignore
    self.processor = CLIPProcessor.from_pretrained(model_name)
    print("\nModel loaded successfully!")
  
  def encode_image(self,image_path):
    image = Image.open(image_path)
    image_input = self.processor(images=image,return_tensors="pt").to(self.device) #type: ignore
    with torch.no_grad():
      image_embedding = self.model.get_image_features(**image_input) #type: ignore
      image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    return image_embedding.cpu().numpy().flatten()
  
  def encode_text(self,query):
    query_input = self.processor(text=query,return_tensors="pt").to(self.device) #type: ignore
    with torch.no_grad():
      query_embedding = self.model.get_text_features(**query_input) #type: ignore
      query_embedding = query_embedding /  query_embedding.norm(dim=-1,keepdim=True)
    return query_embedding.cpu().numpy().flatten()
  
  def encode_batch_images(self,images):
    images = [Image.open(image) for image in images]
    
    
    
