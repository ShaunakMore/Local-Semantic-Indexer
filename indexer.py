import ollama #type:ignore
from shore_db import ShoreDB #type:ignore
from pathlib import Path
from PIL import Image #type:ignore
import gc
import pickle


prompt = "Describe this image briefly and factually in one sentence."       
        
def check_indexing(folder_path):
    folder = Path(folder_path)
    vec_db_path = folder/"index.bin"
    image_map_path = folder/"image_map.pkl"
    
    if vec_db_path.exists() and image_map_path.exists():
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        vec_db = ShoreDB()
        vec_db.load_from_file(vec_db_path)
        with open(image_map_path,"rb") as f:
            image_map = pickle.load(f)
        paths = list(image_map.values())
        missing = [image for image in images if image not in paths]
        if missing:
            print("\nUnindexed images found. Indexing...")
            model_response = captioning(missing,prompt=prompt)
            print(indexing(model_response,missing,vec_db,image_map))
        return vec_db,image_map
    
    else:
        print("\nFolder not indexed. Indexing...")
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        vec_db = ShoreDB()
        image_map = {}
        model_response = captioning(images,prompt=prompt)
        print(indexing(model_response,images,vec_db,image_map))
        vec_db.save_to_disk(str(vec_db_path))
        with open(folder/"image_map.pkl","wb") as f:
            pickle.dump(image_map,f)
        print("\nFolder indexed successfully!")
        return vec_db,image_map
        
            


def captioning(images,prompt):
    responses = []
    for image in images:
        try:
            response = ollama.chat(
                model='llama3.2-vision:11b',
                messages=[{
                    'role': 'user',
                    'content':prompt,
                    'images': [image]
                }]
            )
            caption = response["message"]["content"]
            print("\nDescription by Ollama: \n")
            print(caption)
            responses.append(caption)
        except Exception as e:
            print(f"\nError processing image:{e}")
        finally:
            del image
            gc.collect()
    print("\nImages captioned successfully!\n")
    return responses
    

def indexing(model_responses,image_paths, image_db,image_id_map):
    for (image_path,model_response) in zip(image_paths,model_responses):
        id = image_db.random_id_generator()
        image_id_map[id] = image_path
        model_response_embedding = ollama.embeddings(
            model="nomic-embed-text", 
            prompt=model_response
        )["embedding"]
        image_db.add_vector(id,model_response_embedding)
    return "Image indexed successfully"

def retrieve_images(query,folder,image_db,image_id_map,k):
    folder_path = Path(folder)
    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt = query
    )["embedding"]
    retrieved_images = image_db.k_nearest_neighbours(query_embedding,k)
    for image in retrieved_images:
        image_id = image[0]
        image_path = folder_path/image_id_map[image_id]
        img_read = Image.open(image_path)
        img_read.show()    


db = ShoreDB()

image_id_map = dict()

folder = "D:/Projects/Semantic Image Indexer/Images"

db,image_id_map =check_indexing(folder)

while(True):   
    query = input("\nSearch Image:\n")
    print("\n") 
    retrieve_images(query,folder,db,image_id_map,1)