from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('timeline.html')

@app.route("/embed_graph")
def embed_graph():
    return render_template('embed_visual.html')

@app.route("/hierarchy")
def hierarchy():
    return render_template('hie.html')

if __name__ == '__main__':
    app.run()