from transformers import CLIPProcessor, CLIPModel #type: ignore
import torch
from PIL import Image
from shore_db import ShoreDB # type: ignore
from pathlib import Path
import os
import subprocess
import pickle

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
  
  def encode_batch_images(self,images,batch_size = 32):
    embedding = []
    for i in range(0,len(images),batch_size):
      batch = images[i:i+batch_size]
      image_batch = [Image.open(image) for image in batch]
      image_batch_input = self.processor(images=image_batch,return_tensors="pt").to(device=self.device) #type: ignore
      with torch.no_grad():
        image_embedding = self.model.get_image_features(**image_batch_input) #type: ignore
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        embedding.extend(image_embedding.cpu().numpy())
        del image_batch, image_batch_input, image_embedding
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return embedding     
        
def check_indexing(folder_path):
    folder = Path(folder_path)
    vec_db_path = folder/".index_cache_clip"/"index.bin"
    image_map_path = folder/".index_cache_clip"/"image_map.pkl"
    clip = CLIPImageIndexer(model_name="openai/clip-vit-base-patch16")
    
    if vec_db_path.exists() and image_map_path.exists():
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        vec_db = ShoreDB()
        vec_db.load_from_file(str(vec_db_path))
        with open(image_map_path,"rb") as f:
            image_map = pickle.load(f)
        paths = list(image_map.values())
        missing = [image for image in images if image not in paths]
        if missing:
            print("\nUnindexed images found. Indexing...")
            model_response = image_encoding(missing,clip)
            print(indexing(model_response,missing,vec_db,image_map))
            vec_db.save_to_disk(str(vec_db_path))
            with open(folder/".index_cache_clip"/"image_map.pkl","wb") as f:
                pickle.dump(image_map,f)
        print("\nFolder is Indexed ✅")
        return vec_db,image_map, clip
    else:
        print("\nFolder not indexed. Indexing...")
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        vec_db = ShoreDB()
        image_map = {}
        model_response = image_encoding(images,clip)
        print(indexing(model_response,images,vec_db,image_map))
        hidden_dir = os.path.join(folder, ".index_cache_clip")
        os.makedirs(hidden_dir, exist_ok=True)
        if os.name == "nt":
            subprocess.run(["attrib", "+h", hidden_dir]) 
        vec_db.save_to_disk(str(vec_db_path))
        with open(folder/".index_cache_clip"/"image_map.pkl","wb") as f:
            pickle.dump(image_map,f)
        print("\nFolder indexed successfully!")
        return vec_db,image_map,clip
      
def image_encoding(images,clip):
  try:
      encoded = clip.encode_batch_images(images=images,batch_size=16)
      print("\nImages encoded successfully!\n")
      return encoded
  except Exception as e:
      print(f"\nError processing image:{e}")
      return []
  

def indexing(model_responses,image_paths, image_db,image_id_map):
    for (image_path,model_response) in zip(image_paths,model_responses):
        id = image_db.random_id_generator()
        image_id_map[id] = image_path
        image_db.add_vector(id,model_response.tolist())
    return "Image indexed successfully"
  
def retrieve_images(query,folder,image_db,image_id_map,clip,k):
    folder_path = Path(folder)
    query_embedding = clip.encode_text(query)
    retrieved_images = image_db.k_nearest_neighbours(query_embedding.tolist(),k)
    for image in retrieved_images:
        image_id = image[0]
        image_path = folder_path/image_id_map[image_id]
        img_read = Image.open(image_path)
        img_read.show() 
    
db = ShoreDB()

image_id_map = dict()


folder = "D:/Projects/Semantic Image Indexer/Images"

db,image_id_map,clip = check_indexing(folder)

while(True):   
    query = input("\nSearch Image:\n")
    print("\n") 
    retrieve_images(query,folder,db,image_id_map,clip,1)
    
