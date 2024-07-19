from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import torch
from sklearn.cluster import KMeans
import faiss
import numpy as np

app = Flask(__name__)

# Cargamos el dataset desde el csv
df_courses = pd.read_csv(
    'archive/3.1-data-sheet-udemy-courses-business-courses.csv')

df_courses.rename(columns={
    'course_title': 'name',
    'level': 'level',
    'content_duration': 'duration',
    'subject': 'category'
}, inplace=True)

df_courses['name'] = df_courses['name'].astype(str)

# Cargamos el modelo y tokenizador preentrenados para embeddings
tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained(
    'sentence-transformers/paraphrase-MiniLM-L6-v2')

# Generamos embeddings para los nombres de los cursos
def get_course_embeddings(courses):
    course_embeddings = []
    for course in courses:
        course_name = str(course['name'])
        inputs = tokenizer(course_name, return_tensors='pt',
                           truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            embeddings = model(
                **inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        course_embeddings.append(embeddings)
    return np.array(course_embeddings)

course_embeddings = get_course_embeddings(df_courses.to_dict(orient='records'))
df_courses['embeddings'] = course_embeddings.tolist()

# Agrupamos los cursos en clusters con Kmeans
def cluster_courses(course_embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(course_embeddings)
    return clusters

clusters = cluster_courses(course_embeddings)
df_courses['cluster'] = clusters

# Creamos el FAISS index (una forma rápida de RAG para recuperación de información)
faiss_index = faiss.IndexFlatL2(course_embeddings.shape[1])
faiss_index.add(course_embeddings)

# Configuramos el modelo generativo con GPT2
tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
model_gpt = GPT2LMHeadModel.from_pretrained('gpt2')

# Evaluamos el desempeño del usuario con nuestras reglas
def evaluate_performance(duration, actual_time, exam_score):
    if exam_score >= 85 and actual_time <= duration:
        return "Understands well"
    else:
        return "Needs improvement"

# Recomendación del siguiente curso usando FAISS para recuperación y GPT-2 para generación
def recommend_next_course(course_id, performance, df_courses):
    current_course = df_courses[df_courses['course_id'] == course_id].iloc[0]
    current_cluster = current_course['cluster']

    if performance == "Understands well":
        next_courses = df_courses[(df_courses['cluster'] == current_cluster) &
                                  (df_courses['level'] != current_course['level']) &
                                  (df_courses['level'] == 'Intermediate Level') &
                                  (df_courses['course_id'] != course_id)]
    else:
        next_courses = df_courses[(df_courses['cluster'] == current_cluster) &
                                  (df_courses['level'] == 'Beginner Level') &
                                  (df_courses['course_id'] != course_id)]

    if next_courses.empty:
        return current_course, None, "No suitable course found."

    # Recuperamos el curso más relevante usando FAISS
    next_course_embeddings = np.array(next_courses['embeddings'].tolist())
    query_embedding = np.array(current_course['embeddings']).reshape(1, -1)
    faiss_index_next = faiss.IndexFlatL2(next_course_embeddings.shape[1])
    faiss_index_next.add(next_course_embeddings)
    distances, indices = faiss_index_next.search(query_embedding, 1)
    most_relevant_course = next_courses.iloc[indices[0][0]]

    # Generamos recomendación usando GPT-2
    input_text = (
    f"The user has completed the course '{current_course['name']}' which is about {current_course['category']}.\n"
    f"This course was at the {current_course['level']} and lasted {current_course['duration']} hours.\n"
    f"Based on this, I recommend the next course to be '{most_relevant_course['name']}' which is also about {most_relevant_course['category']}.\n "
    f"This course is at the {most_relevant_course['level']} and will take approximately {most_relevant_course['duration']} hours to complete.")
    inputs = tokenizer_gpt(input_text, return_tensors='pt')
    outputs = model_gpt.generate(
        inputs['input_ids'], max_length=100, max_new_tokens=1, num_return_sequences=1)
    recommendation = tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)

    return current_course, most_relevant_course, recommendation

# Guardamos el progreso del usuario
def save_user_progress(user_progress_df, user_id, course_name, performance, next_course_name, actual_time, exam_score):
    new_entry = pd.DataFrame([{
        'user_id': user_id,
        'completed_course': course_name,
        'performance': performance,
        'next_course': next_course_name if next_course_name else "No suitable course found",
        'actual_time': actual_time,
        'exam_score': exam_score
    }])
    user_progress_df = pd.concat(
        [user_progress_df, new_entry], ignore_index=True)
    user_progress_df.to_csv('user_progress.csv', index=False)
    return user_progress_df

# Aplicación web con Flask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate_user():
    user_id = request.form['user_id']
    course_id = float(request.form['course_id'])
    actual_time = float(request.form['actual_time'])
    exam_score = float(request.form['exam_score'])

    if not user_id or not course_id or actual_time is None or exam_score is None:
        return jsonify({"error": "Missing data"}), 400

    try:
        current_course = df_courses[df_courses['course_id']
                                    == course_id].iloc[0]
        duration = current_course['duration']
    except IndexError:
        return jsonify({"error": "Invalid course_id"}), 400

    performance = evaluate_performance(duration, actual_time, exam_score)
    current_course, next_course, recommendation = recommend_next_course(
        course_id, performance, df_courses)

    # Cargamos el progreso del usuario desde el CSV
    try:
        user_progress_df = pd.read_csv('user_progress.csv')
    except FileNotFoundError:
        user_progress_df = pd.DataFrame(columns=[
                                        'user_id', 'completed_course', 'performance', 'next_course', 'actual_time', 'exam_score'])

    # Guardamos el progreso del usuario
    next_course_name = next_course['name'] if next_course is not None else None
    user_progress_df = save_user_progress(
        user_progress_df, user_id, current_course['name'], performance, next_course_name, actual_time, exam_score)

    if next_course is None:
        return render_template('result.html', current_course=current_course, next_course=None, recommendation=recommendation, performance=performance)

    return render_template('result.html', current_course=current_course, next_course=next_course.to_dict(), recommendation=recommendation, performance=performance)


if __name__ == '__main__':
    app.run(port=5000)
