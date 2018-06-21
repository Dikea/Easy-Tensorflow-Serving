# Easy Tensorflow Serving

Provide easy way to use tensorflow serving. 

## Installation

**Step 1. Install Tensorflow-Serving** <br>

The easy-tensorflow-serving is based on [tensorflow-serving](https://github.com/tensorflow/serving), so install it firstly, see [install instructions](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md).

**Step 2. Install Easy-Tensorflow-Serving** <br>

```sh
git clone git@code.byted.org:huanghailong/easy_tensorflow_serving.git
python setup.py install
```

## How to Use 

There is a simple [example](./example) that show how to use the tool. <br>

**Step 1. Export model** <br>

```sh
cd example
python export_model.py
```

**Step 2. Start server** <br>

```sh
tensorflow_model_server --port=9000 --model_name=test_model --model_base_path=/data00/home/huanghailong/easy_tensorflow_serving/example/tmp/model
```

**Step 3. Build client** <br>

```sh
python predict.py
```

## TODO

- Add flask support.
