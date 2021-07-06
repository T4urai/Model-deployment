from flask import Flask ,render_template,request
import player_scout as ps
from PIL import Image


app = Flask(__name__)


@app.route('/',methods =['GET','POST'])
def player_scouter():
    if request.method=="POST":
        Goals=request.form['Goals']
        player_scout=ps.scout(Goals)
        print(player_scout)

    return render_template('index.html', )

"""

@app.route('/sub',methods =['POST'])
def submit():
    if request.method == "POST":
        name =request.form["username"]
    
    return render_template("sub.html",n= na 
    
    """

if __name__ == "__main__":
    app.run(debug=True)