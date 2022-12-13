# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import streamlit.components.v1 as stc

# Load Udemy Dataset
def load_udemy_csv(data):
    df=pd.read_csv('./udemy_courses.csv')
    return df

# Vectorize the data and get the Cosine Similarirty Matrix
def vectorize_text_to_cosine(data):
    # Vectorize
    cv=CountVectorizer()
    cv_mat=cv.fit_transform(data)
    # Get the Cosine Matrix
    cosine_similarirty_mat=cosine_similarity(cv_mat)
    return cosine_similarirty_mat

# Recommender Function
def get_recommendations(title, cosine_similarirty_mat, df, num_of_recommendation=5):
    # indices of the course
    course_indices=pd.Series(df.index, index=df['course_title']).drop_duplicates()
    # Index of the course
    idx=course_indices[title]

    sim_scores=list(enumerate(cosine_similarirty_mat[idx]))
    sim_scores=sorted(sim_scores, key=lambda x: x[1],reverse=True)
    # We get the sim_scores in JSON format. We will extract the course index from the sim_scores. 
    # The first value will be the entered input itself. Hence, we ignore it by using [1:]
    selected_course_indices= [i[0] for i in sim_scores[1:]]

    result_df = df.iloc[selected_course_indices]
    final_recommended_courses = result_df[['course_title','url','price','num_subscribers']]
    return final_recommended_courses.head(num_of_recommendation)

RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""

def main():
    st.title('Udemy Course Recommender')

    menu=['Home', 'Recommend']
    choice=st.sidebar.selectbox('Menu', menu)

    df=load_udemy_csv('udemy_courses.csv')


    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))
    else:
        st.subheader('Recommend Courses')

        cosine_similarirty_mat=vectorize_text_to_cosine(df['course_title'])
        # get the course input from user from a text input
        search_bar=st.text_input("Enter Course Name or Keyword")
        # Number of Courses to recommend
        num_of_recommendation=st.sidebar.number_input("Number Of Courses", 4, 25, 5)

        if st.button("Recommend"):
            if search_bar is not None:
                result=get_recommendations(search_bar, cosine_similarirty_mat, df, num_of_recommendation)
                
                #st.write(result)
                for row in result.iterrows():
                    rec_title = row[1][0]
                    rec_url = row[1][1]
                    rec_price = row[1][2]
                    rec_subs = row[1][3]

                    # st.write("Title",rec_title,)
                    stc.html(RESULT_TEMP.format(rec_title, rec_url,rec_price,rec_subs),height=350)

if __name__ == '__main__':
    main()

# Run streamlit app
# python -m streamlit run app.py