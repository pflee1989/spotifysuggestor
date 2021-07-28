import pandas as pd
from flask import Flask, render_template, request
from .models import song_model, to_list


df = pd.read_csv('suggestor/edited_data_v2.csv')

def create_app():
    app = Flask(__name__)

    @app.route("/", methods = ["GET", "POST"])
    def main_page():
        """
        1. Asks user for their name.
        2. Returns personalized welcome message.
        3. Moves user to song input.
        """
        if request.method == "GET":
            return render_template('home.html')
        if request.method == "POST":
            return render_template('greet.html', name=request.form.get("name", "you"))
        
    @app.route("/music", methods = ["GET", "POST"]) 
    def input():
        """ 
        Input user's favorite song.
        Return list of recommended songs.
        """
        if request.method == "GET":
            track_artist = to_list(df)
            return render_template('input_song.html', data=track_artist)
        
        if request.method == "POST":
            input = request.form.get("input_song")
            index = df.loc[df.isin([input]).any(axis=1)].index.tolist()
            index = index[0]
            model = song_model(index)
            return render_template('output_song.html', output_song=input, recommended_song=model)
    
    @app.route("/about") 
    def about():
        """ 
        Our about us page.
        """
        return render_template('about.html')
    
    return app