from flask import Flask, jsonify, request
import numpy as np
from src import testSpiral, testWave
app = Flask(__name__)

@app.route("/")
def main():
  return "Parkinson GP"  

@app.route("/wave/<string:url>")
def waveText(url):
  result = testWave.testWaveImage(url)
  return jsonify({'prediction': result.tolist()})


@app.route("/spiral/<string:url>")
def spiralTest(url):
  result = testSpiral.testSpiralImage(url)
  return jsonify({'prediction': result.tolist()})

if __name__ == "__main__":
  app.run()