#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression




# In[7]:


df=pd.read_csv('spotify_millsongdata.csv')


# In[8]:


df.loc[0,'text']


# In[9]:


df.shape


# In[10]:


df=df.sample(25000).drop('link', axis=1).reset_index(drop=True)


# In[11]:


pp=df.copy()


# In[12]:


pp['text'] = pp['text'].apply(lambda x: str(x).lower().replace(r'^\w\s',' ').replace(r'\n',' ',))


# In[13]:


df['text']= df['text'].apply(lambda x :str(x).replace(r'\n\r',' '))


# NOW that we have assigned the link to another column we can drop them
# 

# In[14]:


df.shape


# In[15]:


artist_list=df.groupby('artist').count()


# Now we are having 643 artist list in our dataset
# 

# In[16]:


artist_list.count()


# In[17]:


df['song']=df['song'].str.lower()


# In[18]:


df['text'].str.count('\n').sum()


# In[19]:


df.columns


# In[20]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


# import nltk
# it is a nlp library , it is used to process the text data
# from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
# stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# In this the similar words which are having same meaning but differ in spelling are changed into one common word
# 

# In[22]:


def preprocess_text(text):
    tokens = word_tokenize(text)  # Tokenization and lowercase conversion
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]  # Stopword and punctuation removal
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)
     


# In[23]:


# # import nltk
# # nltk.download('punkt')
# token(df['text'][0])
# token('hello world beautiful beauty')


# In[24]:


df['cleaned text']=df['text'].apply(preprocess_text)


# In[25]:


tdidf=TfidfVectorizer(analyzer='word', stop_words='english')
metrixs=tdidf.fit_transform(df['cleaned text'])


# Here we are taking sample data for testing 
# 

# In[26]:


# from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import csr_matrix

# # Assuming 'matrix' is a sparse matrix in CSR format
# sparse_matrix = csr_matrix(matrix)

# # Compute cosine similarity using sparse matrices
similar = cosine_similarity(metrixs)


# In[27]:


similar[0]


# In[28]:


df['song']


# In[29]:


# df[df['song']=="city's burning"].index[0]


# Now we are making recommder function for getting the songs
# 

# In[30]:


def recommder(song_name):
    idx=df[df['song']==song_name].index[0]
    distance=sorted(list(enumerate(similar[idx])), key=lambda x: x[1], reverse=True)
    song=[]
    for s_id in distance[1:20]:
        song.append(df.iloc[s_id[0]].song)
    return song
    
    


# In[31]:


# recommder("steel drivin' man")


# In[32]:


print(pp.iloc[0].text)


# Now we are gonna do the dumping of our code to bytes through pickle with dump method

# In[33]:


def artist_songs(song_name):
    index_range=pp[pp['song']==song_name].artist
    artist_list=pp[pp['artist']==index_range.iloc[0]].song
    return artist_list


# In[34]:


from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# In[35]:


# del sampled_descriptions


# In[36]:


sampled_descriptions = df


# now we are going to take the sample data

# In[37]:


# # Randomly sample a subset of song descriptions for clustering (adjust sample size as needed)
# sample_size = 1000
# random_indices = np.random.choice(len(df['cleaned text']), size=sample_size, replace=False)
# sampled_descriptions = [df.iloc[i] for i in random_indices]


# In[38]:


sampled_descriptions=pd.DataFrame(sampled_descriptions)


# we need to change it to data frame
# 
# now that we got the sampled data , we gonna do tfidf on that sampled data
# 

# In[39]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
tdif=TfidfVectorizer(analyzer='word', stop_words='english', max_features=100)


# In[40]:


matrix=tdif.fit_transform(sampled_descriptions['cleaned text'])


# In[41]:


print(matrix.shape)


# In[42]:


# del wcss
# del silhouette_scores


# In[43]:


wcss = []
silhouette_scores = []

# Try different values of k (number of clusters) and compute WCSS and silhouette scores
for k in range(2, 25):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
    kmeans.fit(matrix)
    wcss.append(kmeans.inertia_)  # Within-cluster sum of squares
    silhouette_scores.append(silhouette_score(matrix, kmeans.labels_))
    


# In[44]:


# Plotting Elbow Method (WCSS vs. Number of Clusters)
plt.figure(figsize=(10, 5))
plt.plot(range(2, 25), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k (Song Descriptions)')
plt.xticks(np.arange(2, 25))
plt.grid(True)
plt.show()

# Plotting Silhouette Score vs. Number of Clusters
plt.figure(figsize=(10, 5))
plt.plot(range(2, 25), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k (Song Descriptions)')
plt.xticks(np.arange(2, 25))
plt.grid(True)
plt.show()


# In[45]:


from sklearn.decomposition import TruncatedSVD


# Initialize MiniBatchKMeans clustering with optimal k (e.g., based on elbow method or silhouette score)

# In[46]:


import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans

# Assuming 'matrix' and 'tfidf_svd' are defined earlier in your code

svd = TruncatedSVD(n_components=2, random_state=0)
tfidf_svd = svd.fit_transform(matrix)

fig, axs = plt.subplots(5, 4)
fig.set_size_inches(12, 15)

scatter_plots = []

# Initialize MiniBatchKMeans clustering with optimal k
for i in range(2, min(21, 5*4 + 2)):  # Adjust the range
    k = i
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
    kmeans.fit(matrix)

    # Get cluster labels and plot clusters in a scatter plot
    cluster_labels = kmeans.labels_
    scatter_plot = axs[int((i - 2) / 4), (i - 2) % 4].scatter(tfidf_svd[:, 0], tfidf_svd[:, 1], c=cluster_labels, cmap='viridis')

    axs[int((i - 2) / 4), (i - 2) % 4].set_xlabel('1')
    axs[int((i - 2) / 4), (i - 2) % 4].set_ylabel('2')
    axs[int((i - 2) / 4), (i - 2) % 4].set_title(f'clusters ({k})')
    axs[int((i - 2) / 4), (i - 2) % 4].grid(True)

    scatter_plots.append(scatter_plot)  # Add scatter plot object to the list

# Add colorbar
# fig.colorbar(scatter_plots[-1], ax=axs.ravel().tolist(), location='left', pad=0.05, label='Cluster')

plt.tight_layout()
plt.show()


# In[47]:


kmeans = KMeans(n_clusters=18, init='k-means++', random_state=42)
kmeans.fit(matrix)

# Add cluster labels to dataset
sampled_descriptions['cluster'] = kmeans.labels_+2


# In[48]:


set(kmeans.labels_)


# In[49]:


kmeans.cluster_centers_.shape


# In[50]:


import pickle


# In[51]:


# pickle.dump(similar, open('similar.pkl', 'wb'))


# In[52]:


# pickle.dump(df, open('df.pkl', 'wb'))


# In[53]:


# Dictionary for genre values (with values converted to strings)
genre_values = {
    'Classical': ['Orchestral, Symphonic, Compositional complexity, Instrumental mastery, Elegance, '
                 'Formal structure, Timelessness, Emotionally evocative, Virtuosity'],
    'Heavy Metal': ['Aggression, Distortion, Amplification, Dark themes, Intensity, Rebellion, '
                   'Complex guitar riffs, Screaming vocals, Power'],
    'Punk': ['Anarchy, Rebellion, DIY ethos, Raw energy, Fast tempo, Anti-establishment, '
            'Political dissent, Social commentary, Minimalism'],
    'Reggaeton': ['Rhythm, Danceability, Spanish lyrics, Latin beats, Urban culture, Sensuality, '
                 'Party atmosphere, Caribbean influence, Fusion of genres'],
    'Indie': ['Alternative, DIY spirit, Non-conformity, Quirkiness, Artistic expression, Subculture, '
             'Introspection, Authenticity, Underground'],
    'Gospel': ['Faith, Spirituality, Worship, Soulful vocals, Joy, Testimony, Redemption, Church, '
              'Hope'],
    'Metalcore': ['Aggression, Breakdowns, Screamed vocals, Melodic elements, Technical proficiency, '
                 'Emotional intensity, Dual vocal styles, Heavy guitar riffing, Catharsis'],
    'Ska': ['Upbeat tempo, Offbeat rhythm, Horn sections, Jamaican influence, Danceable grooves, Fun, '
           'Lightheartedness, Subcultural identity, Skanking'],
    'Funk': ['Groove, Rhythmic syncopation, Soulful vocals, Danceability, Improvisation, '
            'Tight instrumentation, Syncopated basslines, Horn sections, Party atmosphere'],
    'Disco': ['Dancefloor, Glittering, Funky basslines, Four-on-the-floor beat, Studio production, '
             'High energy, Glamour, Nightlife, Liberation'],
    'Country': ['Rural themes, Simple melodies, Acoustic instruments, Storytelling, '
                'Western influences, Folk traditions, Heartache, Americana, Honesty'],
    'Hip Hop': ['Rap, Beats, Urban culture, Street life, DJ scratching, Sampling, '
                'Rhyme, Flow, Authenticity'],
    'Jazz': ['Improvisation, Swing, Syncopation, Blue notes, Instrumental virtuosity, '
             'Big band, Coolness, Soulfulness, Experimentalism'],
    'Rock': ['Guitars, Rebellion, Power chords, Amplification, Youth culture, '
             'Verses and choruses, Expressiveness, Energy, Attitude'],
    'Electronic': ['Synthesizers, Beats, Sampling, Dancefloor, Futurism, '
                   'Soundscapes, Repetition, Technological innovation, Experimentation'],
    'Blues': ['12-bar structure, Call and response, Soulful vocals, Guitar solos, '
              'Lament, Struggle, Improvisation, Feeling, Authenticity'],
    'R&B': ['Rhythm and blues, Soul, Groove, Sensuality, Vocals, Love themes, '
            'Influential artists, Heartfelt lyrics, Melodic hooks'],
    'Pop': ['Catchy melodies, Hooks, Commercial appeal, Studio production, '
            'Youth culture, Upbeat tempo, Sing-along choruses, Mainstream appeal, Radio-friendly'],
    'Alternative': ['Non-mainstream, Innovation, Diversity, Independent spirit, '
                     'Experimentalism, Subversion, Eclecticism, Counterculture'],
    'World': ['Cultural diversity, Traditional instruments, Ethnic rhythms, '
              'Global fusion, Folklore, International influences, Exoticism, Celebration']
}

# Dictionary for genre labels
genre_labels = {
    'Classical': 0,
    'Heavy Metal': 1,
    'Punk': 2,
    'Reggaeton': 3,
    'Indie': 4,
    'Gospel': 5,
    'Metalcore': 6,
    'Ska': 7,
    'Funk': 8,
    'Disco': 9,
    'Country': 10,
    'Hip Hop': 11,
    'Jazz': 12,
    'Rock': 13,
    'Electronic': 14,
    'Blues': 15,
    'R&B': 16,
    'Pop': 17,
    'Alternative': 18,
    'World': 19
}

genre_themes = {
    'Classical': 'Orchestral, Symphonic, Compositional complexity, Instrumental mastery, Elegance, '
                 'Formal structure, Timelessness, Emotionally evocative, Virtuosity',
    'Heavy Metal': 'Aggression, Distortion, Amplification, Dark themes, Intensity, Rebellion, '
                   'Complex guitar riffs, Screaming vocals, Power',
    'Punk': 'Anarchy, Rebellion, DIY ethos, Raw energy, Fast tempo, Anti-establishment, '
            'Political dissent, Social commentary, Minimalism',
    'Reggaeton': 'Rhythm, Danceability, Spanish lyrics, Latin beats, Urban culture, Sensuality, '
                 'Party atmosphere, Caribbean influence, Fusion of genres',
    'Indie': 'Alternative, DIY spirit, Non-conformity, Quirkiness, Artistic expression, Subculture, '
             'Introspection, Authenticity, Underground',
    'Gospel': 'Faith, Spirituality, Worship, Soulful vocals, Joy, Testimony, Redemption, Church, '
              'Hope',
    'Metalcore': 'Aggression, Breakdowns, Screamed vocals, Melodic elements, Technical proficiency, '
                 'Emotional intensity, Dual vocal styles, Heavy guitar riffing, Catharsis',
    'Ska': 'Upbeat tempo, Offbeat rhythm, Horn sections, Jamaican influence, Danceable grooves, Fun, '
           'Lightheartedness, Subcultural identity, Skanking',
    'Funk': 'Groove, Rhythmic syncopation, Soulful vocals, Danceability, Improvisation, '
            'Tight instrumentation, Syncopated basslines, Horn sections, Party atmosphere',
    'Disco': 'Dancefloor, Glittering, Funky basslines, Four-on-the-floor beat, Studio production, '
             'High energy, Glamour, Nightlife, Liberation',
    'Country': 'Rural themes, Simple melodies, Acoustic instruments, Storytelling, '
                'Western influences, Folk traditions, Heartache, Americana, Honesty',
    'Hip Hop': 'Rap, Beats, Urban culture, Street life, DJ scratching, Sampling, '
                'Rhyme, Flow, Authenticity',
    'Jazz': 'Improvisation, Swing, Syncopation, Blue notes, Instrumental virtuosity, '
             'Big band, Coolness, Soulfulness, Experimentalism',
    'Rock': 'Guitars, Rebellion, Power chords, Amplification, Youth culture, '
}


# In[54]:


kmeans.labels_.shape


# In[55]:


genre_values


# In[56]:


genre_description_list = []

for genre, description_list in genre_values.items():
    for description in description_list:
        genre_description_list.append({'Genre': genre, 'Description': description})

genre_data = pd.DataFrame(genre_description_list)

genre_data


# In[57]:


# del genre


# In[58]:


genre_tfidf_matrix =tdif.fit_transform(genre_themes.values())
cluster_descriptions = tdif.transform(sampled_descriptions['cleaned text']) 
cluster_centroids = kmeans.cluster_centers_  
print(cluster_centroids.shape,genre_tfidf_matrix.shape)


# In[59]:


similarities = cosine_similarity(cluster_centroids, genre_tfidf_matrix)
similarities


# In[60]:


cluster_genre_labels = {}
for i, cluster_similarities in enumerate(similarities):
    most_similar_genre_index = cluster_similarities.argmax()
    cluster_genre_labels[i] = genre_data.loc[most_similar_genre_index, 'Genre']

# Add genre labels to dataset
# data['genre'] = [cluster_genre_labels[cluster] for cluster in kmeans.labels_]


# In[61]:


cluster_genre_labels


# In[62]:


sampled_descriptions['cluster']=[cluster_genre_labels[cluster] for cluster in kmeans.labels_]


# In[63]:


sampled_descriptions


# In[ ]:




