import flask
import io
import string
import time
import os
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return 'SUCCESS: Health Check'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='9090')