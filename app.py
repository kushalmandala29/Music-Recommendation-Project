import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# from model.ipynb import artist_songs
# import df
# import simila 

CLIENT_ID = "6ab005736a2e4aa5aa553948b7f89c5e"
CLIENT_SECRET ="7b5fe373b10044488ccdd9256643191d"


client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp=spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# we need the album id to get the album cover

def get_song_album_cover_url(song_name, artist_name):
    result=sp.search(q=f"track: {song_name} artist:{artist_name}", type='track')
    
    if result and result['tracks']['items']:
        track=result['tracks']['items'][0]
        album_cover_url=track['album']['images'][0]['url']
        print(album_cover_url)
        return album_cover_url
    else:
        return 'https://i.postimg.cc/0QNxYz4V/socia1.%20.png'
    

def recommend(song):
    index = music[music['song'] == song].index[0]
    distance= sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distance[1:6]:
        artist=music.iloc[i[0]].name
        print(artist)
        print(music.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)
    return recommended_music_names, recommended_music_posters

def get_artist_songs(artist_name):
    results = sp.search(q='artist:' + artist_name, type='artist')

    # Check if the artist was found
    if results['artists']['items']:
    # Get the artist ID
        artist_id = results['artists']['items'][0]['id']
    
    # Get the top tracks for the artist
        top_tracks = sp.artist_top_tracks(artist_id)
        track_images=[]
        for track in top_tracks['tracks']:
            if track['album']['images']:
                track_images.append(track['album']['images'][0]['url'])
            else:
                track_images.append('https://i.postimg.cc/0QNxYz4V/socia1.png')
    
    
    else:
        print(f"Artist '{artist_name}' not found.")

st.header('Music Recommender System')
music=pickle.load(open('df.pkl', 'rb'))
similarity=pickle.load(open('similar.pkl', 'rb'))


movie_list=music['song'].values
selected_movie=st.selectbox("Type or select a song from the dropdown", movie_list)


if st.button('Recommend'):
    recommended_music_names, recommended_music_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])
    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])
if st.button('artist_playlist'):
    recommended_music_names, recommended_music_posters = get_artist_songs(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])
    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])
      

        