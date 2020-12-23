
from flask import Flask, request, flash, render_template
from flask import send_from_directory
import analyzing


app = Flask(__name__,template_folder="web")
app.secret_key = "releaf secret"

@app.route('/<path:path>')
def others(path):
    return send_from_directory("web", path)


@app.route('/')
def index():
    return others("index.html")




@app.route('/api', methods=['POST'])
def api():
    if request.method == 'POST':
        # check if the post request has the file part
        if not ('camera'  in request.files or 'gallery'  in request.files):
            return "error"

        file= request.files['gallery']
        if file.filename == '':
            flash('No selected file')
            file = request.files['camera']
            if file.filename == '':
                flash('No selected file')
                return "Please select a file!"


        # if user does not select file, browser also
        # submit an empty part without filename

        if file:
            filename = "uploads/asd.png"
            file.save(filename)

            preds = analyzing.insertImg(filename)

            return render_template("results.html", data=preds)
        return "Only post requests!"



# app.run(host='0.0.0.0', threaded=False, debug=True)
app.run( threaded=False, debug=True)
