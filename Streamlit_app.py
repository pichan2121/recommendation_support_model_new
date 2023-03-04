import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
import pickle
import glob
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
import operator

import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

st.write("### WELCOME TO RECOMMENDATION SUPPORT MODEL FOR E-SHOPPING ! 👋")




STOP_WORD_FILE = 'vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')
     
# load data item
data=pd.read_csv("newdata.csv")

# xulytext
newdata=pd.read_csv("xulytext.csv")



# load data review
data2=pd.read_csv("reviewdata.csv")
# read column
data2=data2[['userId','productId','Rating']]

# read matrix
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data2, test_size =.20, random_state=10)



    
# read ma trix
csr_matrix =  scipy.sparse.load_npz("sparse_matrix.npz")
        

# function popularity collaborative fillering:
class popularity_based_recommender_model():
    def __init__(self, train_data, test_data, user_id, item_id):
        self.train_data = train_data
        self.test_data = test_data
        self.user_id = user_id
        self.item_id = item_id
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def fit(self):
        #Get a count of user_ids for each unique product as recommendation score
        train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
    
        #Sort the products based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a recommendation rank based upon score
        train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data_sort.head(20)

    #Use the popularity based recommender system model to make recommendations
    def recommend(self, user_id, n=5):    
        user_recommendations = self.popularity_recommendations
        
        #Filter products that are not rated by the user
        products_already_rated_by_user = self.train_data[self.train_data[self.user_id] == user_id][self.item_id]        
        user_recommendations = user_recommendations[~user_recommendations[self.item_id].isin(products_already_rated_by_user)]
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols].head(n)     
       
        return user_recommendations
    
    
    def predict_evaluate(self):        
        ratings = pd.DataFrame(self.train_data.groupby(self.item_id)['Rating'].mean())
        
        pred_ratings = [];            
        for data in self.test_data.values:
            if(data[1] in (ratings.index)):
                pred_ratings.append(ratings.loc[data[1]])
            else:
                pred_ratings.append(0)
        
        mse = mean_squared_error(self.test_data['Rating'], pred_ratings)
        rmse = sqrt(mse)
        return rmse



def intro():
    import streamlit as st

    st.subheader("Introduction")
    st.sidebar.success("Select a model to optimize your searching ")

    st.markdown(
        """
         
        A recommendation system (or recommender system) is a class of machine learning that uses data to help predict, narrow down, 
        and find what people are looking for among an exponentially growing number of options.

       **👈 Select a demo from the dropdown on the left** to see some recommended items 
        of what available models were built

        ### Category Model for Recommendation system? """)
    
    st.markdown(
        """
         
       
        ### Content filtering: """)
    
        
    st.markdown(
        """

        - Uses the attributes or features of an item such as item description, group price and rating level to recommend other items similar to the user’s preferences"""  
    )
    
    st.image("data:image/jpeg;base64,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")
    
    st.markdown(
        """

        ### Collaborative filtering :

        - Collaborative filtering algorithms recommend items (this is the filtering part) based on preference information from many users (this is the collaborative part). This approach uses similarity of user preference behavior,  given previous interactions between users and items, recommender algorithms learn to predict future interaction"""  
    )
    
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQLb6qM94DoDyn6Rbz4xRys50Iq--9cDL3-zXdrKE1UCYJgsvhGSjAqb2wFg7CbWR9ixnA&usqp=CAU")
    

    st.markdown(
        """

        ### Popularity Model and Collaborative Filtering:

        - Uses the attributes or features of an item such as item description, group price and rating level to recommend other items similar to the user’s preferences"""  
    )
    
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS_OvF4T5Yd66AJDDY-ioFGEtZEk9BDw64MzA&usqp=CAU")

    

            
        
            

def content_model():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.neighbors import NearestNeighbors
    import scipy.sparse as sparse
    from sklearn.metrics.pairwise import cosine_similarity
    import operator
    from joblib import Parallel, delayed
    import joblib
    import scipy.sparse
    import numpy as np
    import streamlit as st
    import streamlit.components.v1 as components
    from PIL import Image
    from urllib.error import URLError
    tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words=stop_words)
    tfidf_matrix = tf.fit_transform(newdata.tag1)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    results = {}          
    for idx, row in newdata.iterrows():    
                    similar_indices = cosine_similarities[idx].argsort()[-4:-1]
                    similar_items = [(cosine_similarities[idx][i]) for i in similar_indices]
                    similar_items = [(cosine_similarities[idx][i], newdata.index[i]) for i in similar_indices]
                    print(similar_items[0:])
                    results[idx] = similar_items[0:]  
    
    st.write("### Which item are you looking for") 
      
    item=st.text_input('Key in the name of item ex: mouse, earphone, television.ect.')
    clicked= st.button("Search")   
    clicked="Enter"   
        
   
    if  len(item)!=0:  
            with st.container():
                st.write("#### Here is the list of your searching") 
                item.lower()
                item_index=data.loc[data['name'].str.lower().str.startswith(item)]['name'].tolist()
                option =st.selectbox("Select an Item", item_index)
                item_index_selected=data.loc[data['name']==option].index[0]
                st.write("#### Your selection:")
                st.image(data.iloc[item_index_selected,12],use_column_width=True)
                st.write(data.iloc[item_index_selected,4])
                st.write(data.iloc[item_index_selected,6])
                st.write(data.iloc[item_index_selected,9])
                

                # xem 3 sản phẩm tương quan nhất
                st.title("Other Items referred to your selection!")
                item_list=results[item_index_selected]
                
                col1,col2,col3=st.columns(3)
                                
                        
                with col1:
                        st.image(data.iloc[(item_list[0][1]),12],use_column_width=True)
                        st.write(data.iloc[(item_list[0][1]),4])
                        st.write(data.iloc[(item_list[0][1]),6])
                        st.write(data.iloc[(item_list[0][1]),9])
           
                with col2:
                        st.image(data.iloc[(item_list[1][1]),12],use_column_width=True)
                        st.write(data.iloc[(item_list[1][1]),4])
                        st.write(data.iloc[(item_list[1][1]),6])
                        st.write(data.iloc[(item_list[1][1]),9])
           
                        
                with col3:
                        st.image(data.iloc[(item_list[2][1]),12],use_column_width=True)
                        st.write(data.iloc[(item_list[2][1]),4])
                        st.write(data.iloc[(item_list[2][1]),6])
                        st.write(data.iloc[(item_list[2][1]),9])

                

    
def popular_model():
    import streamlit as st
    import time
    import numpy as np

    with st.container():  
        

            st.write("### Which item are you looking for") 
             
            item=st.text_input('Key in your ID')
            clicked= st.button("Search")   
            clicked="Enter"  
            
    if len(item)!=0:
        st.write("### 1. Here are some recommended items based on the most popularity")   
        pr = popularity_based_recommender_model(train_data=train_data, test_data=test_data, user_id='userId', item_id='productId')
        pr.fit()
        list_products=pr.recommend(item)
        lst=list_products['productId'].tolist()      
        col1, col2 , col3, col4, col5= st.columns(5)
     
                      
        for item_check in lst:
                    i=data.loc[data['item_id']==item_check].index[0]
                    
                    if lst.index(item_check)==0:
                        with col1:
                            st.image(data.iloc[i,12],use_column_width=True)
                            st.write(data.iloc[i,4])
                            st.write(data.iloc[i,6])
                            st.write(data.iloc[i,9])
                    elif lst.index(item_check)==1:
                        with col2:
                            st.image(data.iloc[i,12],use_column_width=True)
                            st.write(data.iloc[i,4])
                            st.write(data.iloc[i,6])
                            st.write(data.iloc[i,9])
                    elif lst.index(item_check)==2:
                        with col3:
                            st.image(data.iloc[i,12],use_column_width=True)
                            st.write(data.iloc[i,4])
                            st.write(data.iloc[i,6])
                            st.write(data.iloc[i,9])
                
                    elif lst.index(item_check)==3:
                        with col4:
                            st.image(data.iloc[i,12],use_column_width=True)
                            st.write(data.iloc[i,4])
                            st.write(data.iloc[i,6])
                            st.write(data.iloc[i,9])

                    else:
                        with col5:
                            st.image(data.iloc[i,12],use_column_width=True)
                            st.write(data.iloc[i,4])
                            st.write(data.iloc[i,6])
                            st.write(data.iloc[i,9])

def knn_model():
    import streamlit as st
    import pandas as pd
    import altair as alt
    from joblib import Parallel, delayed
    import joblib
    from urllib.error import URLError
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.neighbors import NearestNeighbors
    import scipy.sparse as sparse
    from sklearn.metrics.pairwise import cosine_similarity
    import operator
    with st.container():
            st.write("### Some recommended items reference to other User's rating which is similar with your rating history:") 
              
            item=st.text_input('Key in your ID')
            
            clicked= st.button("Search")   
            clicked="Enter"  
    
    if len(item)!=0:
           
            user_id = item
            k = 5
            model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k)
            model.fit(csr_matrix)
            distances, indices = model.kneighbors(csr_matrix.getrow(user_id), n_neighbors=5+1)

            # Find the indices of the k most similar users to user_id
            distances, indices = model.kneighbors(csr_matrix.getrow(user_id), n_neighbors=5+1)

            # Get the indices of the products that the k most similar users have rated
            product_indices = csr_matrix[indices.flatten()[1:], :].nonzero()[1]

            # Get the unique product IDs from the product_indices
            product_ids = np.unique(product_indices)
              
                                 
            for n in product_ids:
                j=data.loc[data['item_id']==n].index[0]
                
                st.write("#### Name of Item:",data.iloc[j,4])
                st.write("#### Rating:", data.iloc[j,6])
                st.write("#### Brand:",data.iloc[j,9])
                st.image(data.iloc[j,12],use_column_width=True)
    
                
    

   
                

page_names_to_funcs = {
    "HomePage": intro,
    "Personalized Items[Content Based]": content_model,
    "Reference Recommended Items[Collaborative Filtering]": knn_model,
    "Recommended Items[Popularity Collaborative Filtering]": popular_model
   
}

demo_name = st.sidebar.selectbox("Menu:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()