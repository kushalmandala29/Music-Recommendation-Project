import pickle
import streamlit as st
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


#Enter your client ID here
CLIENT_ID= os.environ.get('CLIENT_ID')

#Enter your client secret here
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_artist_songs(artist_id):
    # Get the artist's top tracks
    results = sp.artist_top_tracks(artist_id)
    
    # Extract relevant information for each track
    artist_songs = []
    for track in results['tracks']:
        song_name = track['name']
        song_cover = track['album']['images'][0]['url'] if track['album']['images'] else None
        spotify_uri = track['external_urls']['spotify'] if track['external_urls'] else None
        artist_songs.append({'song_name': song_name, 'song_cover': song_cover, 'spotify_uri': spotify_uri})
    
    return artist_songs

# Function to get the artist ID from a song name
def get_artist_id(song_name):
    # Search for the song using its name
    results = sp.search(q=song_name, limit=1, type='track')
    
    # Check if any tracks were found
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        
        # Get the first artist's ID
        artist_id = track['artists'][0]['id']
        return artist_id
    else:
        return None
def get_song_album_cover_url(song_name):

    results = sp.search(q=song_name, limit=1, type='track')

    # Check if any tracks were found
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        album_cover_url = track['album']['images'][0]['url'] if track['album']['images'] else None
        spotify_uri = track['external_urls']['spotify'] if track['external_urls'] else None
        return album_cover_url, spotify_uri
    else:
        return 'https://i.postimg.cc/0QNxYz4V/socia1.%20.png'

# def get_song_album_cover_url(song_name, artist_name):
#     result=sp.search(q=f"track: {song_name} artist:{artist_name}", type='track')
    
#     if result and result['tracks']['items']:
#         track=result['tracks']['items'][0]
#         album_cover_url=track['album']['images'][0]['url']
#         # print(album_cover_url)
#         return album_cover_url
#     else:
#         return 'https://i.postimg.cc/0QNxYz4V/socia1.%20.png'

def recommend(song):
    genre='Disco'
    index = music[music['song'] == song].index[0]
    # gg=
    # similarity=
    # similarity= 
    distance= sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distance[1:6]:
        artist=music.iloc[i[0]].name
        # print(artist)
        # print(music.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song))
        recommended_music_names.append(music.iloc[i[0]].song)
        print(recommended_music_names, recommended_music_posters)
    return recommended_music_names, recommended_music_posters

def recommend(song):
    genre='Disco'
    index = music[music['song'] == song].index[0]
    
    distance= sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distance[1:6]:
        artist=music.iloc[i[0]].name
        # print(artist)
        # print(music.iloc[i[0]].song)
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song))
        recommended_music_names.append(music.iloc[i[0]].song)
        print(recommended_music_names, recommended_music_posters)
    return recommended_music_names, recommended_music_posters

# Other functions like recommend(), get_artist_songs(), and get_song_album_cover_url() remain unchanged

st.header('Music Recommender System')
music = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similar.pkl', 'rb'))

song_list = music['song'].values
selected_movie=st.selectbox("Type or select a song from the dropdown", song_list)

# user_input = st.text_input('Enter Your Name')

if st.button('Recommend'):
    recommended_music_names, recommended_music_posters = recommend(selected_movie)
    print(recommended_music_names)
    for i, name in enumerate(recommended_music_names):
        st.text(name)
        # st.image(recommended_music_posters[i][0])
        st.markdown(f"[![Album Cover]({recommended_music_posters[i][0]})]({recommended_music_posters[i][1]})")

st.title('Artist Songs and Covers')

# Input field for song name
song_name = st.text_input('Enter Song Name')

if song_name:
    try:
        # Get the artist ID from the song name
        artist_id = get_artist_id(song_name)
        if artist_id:
            # Get the artist's songs and cover images
            artist_songs = get_artist_songs(artist_id)

            
            st.subheader(f"Songs by the artist based on '{song_name}':")
            for song_info in artist_songs:
                if song_info['song_name']:
                    st.markdown(f"Song Name: {song_info['song_name']}")
                if song_info['song_cover']:
                    
                    st.markdown(f"[![Album Cover]({song_info['song_cover']})]({song_info['spotify_uri']})")

                    
                else:
                    st.markdown(f"[![Album Cover]({song_info['song_cover']}]({'https://i.postimg.cc/0QNxYz4V/socia1.%20.png'})")
                
        else:
            st.warning('No artist found for the given song name')         
                
       
    except Exception as e:
        st.error(f'Error: {e}')


