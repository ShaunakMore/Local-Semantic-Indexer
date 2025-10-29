import ollama #type:ignore
from shore_db import ShoreDB #type:ignore
from pathlib import Path
from PIL import Image #type:ignore
import gc
import pickle


prompt = "Describe this image briefly and factually in one sentence."       
        

image_arr = ["Images/img1.jpg","Images/img2.jpg","Images/img3.jpg","Images/img4.jpg","Images/img5.jpg","Images/img6.jpg"]

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
    return "Image added successfully"

def retrieve_images(query,image_db,image_id_map,k):
    query_embedding = ollama.embeddings(
        model="nomic-embed-text",
        prompt = query
    )["embedding"]
    retrieved_images = image_db.k_nearest_neighbours(query_embedding,k)
    for image in retrieved_images:
        image_id = image[0]
        image_path = image_id_map[image_id]
        img_read = Image.open(image_path)
        img_read.show()    


db = ShoreDB()

image_id_map = dict()

model_response = captioning(image_arr,prompt)
print(indexing(model_response,image_arr,db,image_id_map))



while(True):   
    query = input("\nSearch Image:\n")
    print("\n") 
    retrieve_images(query,db,image_id_map,1)