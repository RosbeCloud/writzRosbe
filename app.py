 # Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 
import streamlit as st
from PIL import Image
# import torch
# from transformers import pipeline
# device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel


# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 

# @st.cache_resource
# def model_translator_en_es():
#     translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
#     return translator

# @st.cache_resource 
# def model_translator_es_en():
#     translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
#     return translator

# @st.cache_resource
# def model_sentiment():
#     classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#     return classifier

# Fxn
# Vectorize + Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	st.write(cosine_sim_mat)
	return cosine_sim_mat



# Recommendation Sys
@st.cache_resource
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the course
	course_indices = pd.Series(df.index,index=df['ad_creative_body']).drop_duplicates()
	# Index of course
	idx = course_indices[title]
#En este código, hay un grafo representado por una matriz de similitud coseno con los enlaces o índices de los cursos dados en el DataFrame.
# Esta matriz de similitud coseno se utiliza para calcular la similaridad entre los cursos y, en base a ello, recomendar la lista de top 10 mejores cursos relacionados. 
# Los resultados se devuelven en una tabla con las columnas'ad_creative_body', 'similarity_score', 'page_name', 'diferencia' y 'sentimiento', y se enumeran según el puntaje de similitud.

	# Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[idx]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[0] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['ad_creative_body','similarity_score','page_name','diferencia','sentimiento']]
	return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #530303;
  border-left: 5px solid #670C0C;">
<h4 style="color:white;">{}</h4>
<p style="color:white;"><span style="color:#F39E70">📈 Score::</span>{}</p>
<p style="color:white;"><span style="color:#F39E70;">🏆	Empresa: </span>{}</p>
<p style="color:white;"><span style="color:#F39E70;">🌞	Días publicado: </span>{}</p>
<p style="color:white;"><span style="color:#F39E70;">🌝 Sentimiento :</span>{}</p>


</div>
"""

# Search For Course 
@st.cache_resource
def search_term_if_not_found(term,df):
	result_df = df[df['ad_creative_body'].str.contains(term)]
	return result_df


def main():

	st.title("Aplicación de copys de redes sociales")
	menu = ["Inicio","Recomendador","Analisis de sentimientos","About",]
 
	
	logo = Image.open('img/osbe.png')
	
	st.sidebar.image(logo, width=300)
 
	with st.sidebar.header("Menu"):
		
		choice = st.selectbox("Menu",menu)

	df = load_data("data/sentimientos_Final.csv")

        
	if choice == "Inicio":
		st.subheader("Inicio")
		st.dataframe(df.sample(5))


	elif choice == "Recomendador":
		st.subheader("Clasificador de copys de redes sociales")
		cosine_sim_mat = vectorize_text_to_cosine_mat(df['ad_creative_body'])
		search_term = st.text_input("Search")
		num_of_rec = st.sidebar.number_input("Numero de resultados",5,30,7)
		if st.button("Buscar"):
			if search_term is not None:
				try:
					results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
					with st.expander("Results as JSON"):
						results_json = results.to_dict('index')
						st.write(results_json)

					for row in results.iterrows():
						rec_title = row[1][0]
						rec_score = row[1][1]
						rec_page_name = row[1][2]
						rec_diferencia = row[1][3]
						rec_num_sub = row[1][4]
						
						
						# st.write("Title",rec_title,)
						stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_page_name,rec_diferencia,rec_num_sub),height=350)
				except:
					results= "Not Found"
					st.warning(results)
					st.info("Suggested Options include")
					result_df = search_term_if_not_found(search_term,df)
					st.dataframe(result_df)

	elif choice == "Analisis de sentimientos":
		st.subheader("Analisis de sentimientos")
		
		sentimiento = st.text_input("Inserta tu texto")


	else:
		st.subheader("Sobre esta aplicación")
		st.text("Esta aplicación cumple con funciones de recomendación de copys de redes sociales")
		st.text("Esta aplicación fue creada por Emmanuel López Navarrete")

if __name__ == '__main__':
	main()


