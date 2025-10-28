import ollama #type:ignore
from shore_db import ShoreDB #type:ignore
from PIL import Image #type:ignore

def indexing(model_response,image_path, image_db,image_id_map):
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
    

prompt = (
    "Look at the image and describe it in one short, factual sentence. "
    "Only mention the main visible objects and their spatial relationship. "
    "Do not include emotions, moods, or details like lighting or atmosphere. "
    "Do not use words like 'appears', 'seems', or 'suggests'. "
    "Format your response like this example: 'a coffee cup on a wooden desk with a laptop'. "
    "Output only the description."
)

response = ollama.chat(
    model='llama3.2-vision:11b',
    messages=[{
        'role': 'user',
        'content':prompt,
        'images': ['img.png']
    }]
)

print("Description by Ollama: \n")
print(response["message"]["content"]+"\n")

model_response = response["message"]["content"]
print(f"\nModel response dtype: {type(model_response)}")

db = ShoreDB()

image_id_map = dict()

print(indexing(model_response,"img.png",db,image_id_map))

query = input("\nSearch Image:\n")
print("\n")

retrieve_images(query,db,image_id_map,1)