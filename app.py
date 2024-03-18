from flask import Flask, render_template, request
import os
import PyPDF2
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import openai

app = Flask(__name__)

# Папка для зберігання завантажених PDF-файлів
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ваш ключ API OpenAI
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
openai.api_key = OPENAI_API_KEY

# Функція для індексації та збереження векторної БД
def index_documents(pdf_file):
    documents = []
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text = page.extractText()
            documents.append(TaggedDocument(text, [page_num]))
    
    # Зберігання векторної БД
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
    model.save(get_tmpfile("doc2vec.model"))

# Функція для отримання відповіді від OpenAI API
def get_answer(question, documents):
    response = openai.Completion.create(
        engine="davinci",  # Можете обрати інший тип моделі, якщо потрібно
        prompt=question,
        documents=documents,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')

# Завантаження файлу та індексація
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            index_documents(filename)
            return 'Файл успішно завантажено та індексовано.'

# Запити користувача
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        model = Doc2Vec.load(get_tmpfile("doc2vec.model"))
        # Здійснення пошуку
        similar_docs = model.docvecs.most_similar(positive=[model.infer_vector(query)], topn=5)
        documents = [document[0] for document in similar_docs]
        # Формування запиту до OpenAI API
        question = query + "\nAnswer:"
        answer = get_answer(question, documents)
        return render_template('results.html', query=query, answer=answer)
    return 'Введіть запит для пошуку.'

if __name__ == '__main__':
    app.run(debug=True)
