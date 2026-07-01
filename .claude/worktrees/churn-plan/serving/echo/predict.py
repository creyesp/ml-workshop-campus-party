import requests


if __name__ == '__main__':
    payload = {
        "inputs": [
            {
                "name": "parameters-np",
                "datatype": "INT32",
                "shape": [2, 2],
                "data": [1, 2, 3, 4],
                "parameters": {
                    "content_type": "np"
                }
            },
            {
                "name": "parameters-str",
                "datatype": "BYTES",
                "shape": [11],
                "data": "hello world 😁",
                "parameters": {
                    "content_type": "str"
                }
            }
        ]
    }

    response = requests.post(
        "http://localhost:8080/v2/models/content-type-example/infer",
        json=payload
    )
    print(response)