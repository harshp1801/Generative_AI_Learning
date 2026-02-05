import json
import numpy as np
import tflite_runtime.interpreter as tflite

Model_path = "simple_rnn_imdb.tflite"
word_index_path = "word_index.json"
max_len = 500

with open(word_index_path,"r") as f:
    word_index = json.load(f)

interpreter = tflite.Interpreter(model_path = Model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(text):
    processed_txt = text.lower().split()
    tokens = [word_index.get(word,2)+3 for word in processed_txt]
    padded = np.zeroes((1,max_len),dtype = np.float32)
    padded[0,-len(tokens):] = tokens[-max_len:]
    return padded

def lambda_handler(event,context):
    body = json.loads(event.get("body","{}"))
    review = body.get("review")

    if not review:
        return {
            "statusCode":400,
            "body":json.dump({
                "error":"Review text missing"
            })
        }
    input_data = preprocess(review)
    interpreter.set_tensor(input_details[0]["index"],input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]
    sentiment = "Positive" if prediciton>0.5 else "Negative"
    return{
        "statusCode":200,
        "body":json.dumps({
            "review":review,
            "sentiment":sentiment,
            "confidence":float(prediction)
        })
    }






