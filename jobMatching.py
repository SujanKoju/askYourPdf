from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from numpy import dot

model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

worker_embeddings=model.encode("experienced full stack engineer with 5 years of front and backendexperience")
job_embeddings=model.encode("looking for a java engineer with experience in backend development")


similarity=dot(worker_embeddings, job_embeddings) / (norm(worker_embeddings) * norm(job_embeddings))
print("Worker Embeddings : " +str(worker_embeddings))
print("Job Embeddings : " +str(job_embeddings))
print("Similarty : " +str(similarity*100) + "%")