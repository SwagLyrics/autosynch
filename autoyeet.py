from flask import Flask, request, send_from_directory, abort
from werkzeug.utils import secure_filename
from autosynch.align import line_align
from autosynch.config import timestamps_dir
from autosynch.playback import mp3_to_wav
from dotenv import load_dotenv
import os

project_folder = os.path.dirname(os.path.abspath(__file__))  # adjust as appropriate
load_dotenv(os.path.join(project_folder, '.env'))

app = Flask(__name__)

passwd = os.environ['PASSWD']

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inputs')
ALLOWED_EXTENSIONS = ['wav', 'flac', 'ogg', 'mp3']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "Hello, this should be fun.\n"


@app.route("/yeet", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        auth = request.form['auth']
        print('this method posted to') 
        if auth != passwd:
            return 'nope, mate.\n'
        return 'you yeeted correct passwd\n'
    return "this is a test\n"


@app.route("/align", methods=["POST"])
def align_file():
    if request.method == "POST":
        auth = request.form['auth']
        if auth != passwd:
            abort(403)
        if 'file' not in request.files:
            return 'give file m8'
        song = request.form['song']
        artist = request.form['artist']
        file = request.files['file']
        if file and allowed_file(file.filename):
            path =os.path.join(INPUT_DIR, secure_filename(file.filename))
            file.save(path)
            print(f'processing {song} by {artist}, file is {secure_filename(file.filename)}')            
            if os.path.splitext(path)[1] == '.mp3':
                path = mp3_to_wav(path)
                print(f'converted mp3 to wav, {path}.')
            print('beginning alignment.')
            line_align({'song':song, 'artist':artist, 'path': path}, dump_dir=timestamps_dir)
            print('finished alignment.')
            aligned_file = f'{artist}_{song}.yml'.replace(' ', '')
            return send_from_directory(timestamps_dir, aligned_file, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
