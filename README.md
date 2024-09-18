
# Music Recommendation System

## Project Overview
This project implements a music recommendation system using text processing techniques (TF-IDF) and K-means clustering. The system analyzes song lyrics or descriptions to group similar songs together and provide personalized music recommendations.



## Features

- Text processing using TF-IDF (Term Frequency-Inverse Document Frequency)
- K-means clustering for grouping similar songs
- Evaluation metrics including Silhouette Score
- Elbow Method for determining optimal number of clusters



## Installation

Clone the repository 

```bash
  https://github.com/kushalmandala29/Music-Recommendation-Project.git
  cd Music-Recommendation-Project
  
```
Create a virtual enviroment 
```bash
  python -m venv myenv
```
Activate venv

* On Windows
```bash
  myenv\Scripts\activate

```

* On Linux
```bash
  source myenv/bin/activate
```

Installation of all Packages
```bash
  pip install -r requirements.txt

```




## Environment Variables

To run this project, you will need to add the following environment variables

```bash
 Export CLIENT_ID=<your CLIENT_ID>
 Export CLIENT_SECRET=<your secreat key>
  
```

Run the model
```bash
jupyter nbconvert --to script model.ipynb
python model.py
```
Run the script
```bash
streamlit run app.py
```


