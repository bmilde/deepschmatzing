# low tec job scheduler, over http
import collections

from flask import Flask
app = Flask(__name__)

jobs = {



}

predictions = {}
epoch_info = {}
weight_info = {}

@app.route('/getjob')
def hello_world():
    return 'Hello World!'

@app.route('update_epoch')
def update_epoch(myid):
    epoch_info[myid] = 

if __name__ == '__main__':
    app.run()

