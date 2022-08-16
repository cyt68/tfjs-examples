/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

// This is a helper class for loading and managing MNIST data specifically.
// It is a useful example of how you could create your own data manager class
// for arbitrary data though. It's worth a look :)
import { IMAGE_H, IMAGE_W, MnistData } from './data';
import char_set from './char_set.json';

// This is a helper class for drawing loss graphs and MNIST images to the
// window. For the purposes of understanding the machine learning bits, you can
// largely ignore it
import * as ui from './ui';

/**
 * Creates a convolutional neural network (Convnet) for the MNIST data.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
function createConvModel() {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  const model = tf.sequential();

  // The first layer of the convolutional neural network plays a dual role:
  // it is both the input layer of the neural network and a layer that performs
  // the first convolution operation on the input. It receives the 28x28 pixels
  // black and white images. This input layer uses 16 filters with a kernel size
  // of 5 pixels each. It uses a simple RELU activation function which pretty
  // much just looks like this: __/
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_H, IMAGE_W, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  // After the first layer we include a MaxPooling layer. This acts as a sort of
  // downsampling using max values in a region instead of averaging.
  // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Our third layer is another convolution, this time with 32 filters.
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));

  // Max pooling again.
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

  // Add another conv2d layer.
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
  // represent numbers, but it's the same idea if you had classes that
  // represented other entities like dogs and cats (two output classes: 0, 1).
  // We use the softmax function as the activation for the output layer as it
  // creates a probability distribution over our 10 classes so their output
  // values sum to 1.
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  return model;
}

/**
 * Creates a model consisting of only flatten, dense and dropout layers.
 *
 * The model create here has approximately the same number of parameters
 * (~31k) as the convnet created by `createConvModel()`, but is
 * expected to show a significantly worse accuracy after training, due to the
 * fact that it doesn't utilize the spatial information as the convnet does.
 *
 * This is for comparison with the convolutional network above.
 *
 * @returns {tf.Model} An instance of tf.Model.
 */
function createDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [IMAGE_H, IMAGE_W, 1] }));
  model.add(tf.layers.dense({ units: 42, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  return model;
}

/**
 * This callback type is used by the `train` function for insertion into
 * the model.fit callback loop.
 *
 * @callback onIterationCallback
 * @param {string} eventType Selector for which type of event to fire on.
 * @param {number} batchOrEpochNumber The current epoch / batch number
 * @param {tf.Logs} logs Logs to append to
 */

/**
 * Compile and train the given model.
 *
 * @param {tf.Model} model The model to train.
 * @param {onIterationCallback} onIteration A callback to execute every 10
 *     batches & epoch end.
 */
async function train(model, onIteration) {
  ui.logStatus('Training model...');

  // Now that we've defined our model, we will define our optimizer. The
  // optimizer will be used to optimize our model's weight values during
  // training so that we can decrease our training loss and increase our
  // classification accuracy.

  // We are using rmsprop as our optimizer.
  // An optimizer is an iterative method for minimizing an loss function.
  // It tries to find the minimum of our loss function with respect to the
  // model's weight parameters.
  const optimizer = 'rmsprop';

  // We compile our model by specifying an optimizer, a loss function, and a
  // list of metrics that we will use for model evaluation. Here we're using a
  // categorical crossentropy loss, the standard choice for a multi-class
  // classification problem like MNIST digits.
  // The categorical crossentropy loss is differentiable and hence makes
  // model training possible. But it is not amenable to easy interpretation
  // by a human. This is why we include a "metric", namely accuracy, which is
  // simply a measure of how many of the examples are classified correctly.
  // This metric is not differentiable and hence cannot be used as the loss
  // function of the model.
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Batch size is another important hyperparameter. It defines the number of
  // examples we group together, or batch, between updates to the model's
  // weights during training. A value that is too low will update weights using
  // too few examples and will not generalize well. Larger batch sizes require
  // more memory resources and aren't guaranteed to perform better.
  const batchSize = 320;

  // Leave out the last 15% of the training data for validation, to monitor
  // overfitting during training.
  const validationSplit = 0.15;

  // Get number of training epochs from the UI.
  const trainEpochs = ui.getTrainEpochs();

  // We'll keep a buffer of loss and accuracy values over time.
  let trainBatchCount = 0;

  const trainData = data.getTrainData();
  const testData = data.getTestData();

  const totalNumBatches =
    Math.ceil(trainData.xs.shape[0] * (1 - validationSplit) / batchSize) *
    trainEpochs;

  // During the long-running fit() call for model training, we include
  // callbacks, so that we can plot the loss and accuracy values in the page
  // as the training progresses.
  let valAcc;
  await model.fit(trainData.xs, trainData.labels, {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainBatchCount++;
        ui.logStatus(
          `Training... (` +
          `${(trainBatchCount / totalNumBatches * 100).toFixed(1)}%` +
          ` complete). To stop training, refresh or close page.`);
        ui.plotLoss(trainBatchCount, logs.loss, 'train');
        ui.plotAccuracy(trainBatchCount, logs.acc, 'train');
        if (onIteration && batch % 10 === 0) {
          onIteration('onBatchEnd', batch, logs);
        }
        await tf.nextFrame();
      },
      onEpochEnd: async (epoch, logs) => {
        valAcc = logs.val_acc;
        ui.plotLoss(trainBatchCount, logs.val_loss, 'validation');
        ui.plotAccuracy(trainBatchCount, logs.val_acc, 'validation');
        if (onIteration) {
          onIteration('onEpochEnd', epoch, logs);
        }
        await tf.nextFrame();
      }
    }
  });

  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = testResult[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  ui.logStatus(
    `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
    `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
}

/**
 * Show predictions on a number of test examples.
 *
 * @param {tf.Model} model The model to be used for making the predictions.
 */
async function showPredictions(model) {
  const testExamples = 100;
  const examples = data.getTestData(testExamples);

  // Code wrapped in a tf.tidy() function callback will have their tensors freed
  // from GPU memory after execution without having to call dispose().
  // The tf.tidy callback runs synchronously.
  tf.tidy(() => {
    const output = model.predict(examples.xs);

    // tf.argMax() returns the indices of the maximum values in the tensor along
    // a specific axis. Categorical classification tasks like this one often
    // represent classes as one-hot vectors. One-hot vectors are 1D vectors with
    // one element for each output class. All values in the vector are 0
    // except for one, which has a value of 1 (e.g. [0, 0, 0, 1, 0]). The
    // output from model.predict() will be a probability distribution, so we use
    // argMax to get the index of the vector element that has the highest
    // probability. This is our prediction.
    // (e.g. argmax([0.07, 0.1, 0.03, 0.75, 0.05]) == 3)
    // dataSync() synchronously downloads the tf.tensor values from the GPU so
    // that we can use them in our normal CPU JavaScript code
    // (for a non-blocking version of this function, use data()).
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
}

function createModel() {
  let model;
  const modelType = ui.getModelTypeId();
  if (modelType === 'ConvNet') {
    model = createConvModel();
  } else if (modelType === 'DenseNet') {
    model = createDenseModel();
  } else {
    throw new Error(`Invalid model type: ${modelType}`);
  }
  return model;
}

let data;
async function load(status) {
  data = new MnistData();
  await data.load();
  if (!status) {
    return
  }
  await data.loadImg();
}

// This is our main function. It loads the MNIST data, trains the model, and
// then shows what the model predicted on unseen test data.
ui.setTrainButtonCallback(async () => {
  ui.logStatus('Loading MNIST data...');
  await load();

  ui.logStatus('Creating model...');
  const model = createModel();
  model.summary();

  ui.logStatus('Starting model training...');
  await train(model, () => showPredictions(model));
});

ui.setTestCallback(async () => {
  ui.logStatus('Loading test data...');
  await load(true);

  ui.logStatus('Loading model...');
  const model = await tf.loadLayersModel('http://localhost:8081/trainedModel/model.json');

  ui.logStatus('Starting model training...');
  tf.tidy(() => {
    const testExamples = 1;
    const examples = data.getTestData(testExamples, true);
    const output = model.predict(examples.xs);
    const axis = 1;
    const labels = Array.from(examples.labels.argMax(axis).dataSync());
    const predictions = Array.from(output.argMax(axis).dataSync());

    ui.showTestResults(examples, predictions, labels);
  });
});

var MAX_HEIGHT = 100;
// 渲染
function render(src) {
  // 创建一个 Image 对象
  var image = new Image();
  // 绑定 load 事件处理器，加载完成后执行
  image.onload = function () {
    // 获取 canvas DOM 对象
    // var canvas = document.getElementById("myCanvas");
    const canvas = document.createElement('canvas');
    // 如果高度超标
    if (image.height > MAX_HEIGHT) {
      // 宽度等比例缩放 *=
      image.width *= MAX_HEIGHT / image.height;
      image.height = MAX_HEIGHT;
    }
    // 获取 canvas的 2d 环境对象,
    // 可以理解Context是管理员，canvas是房子
    var ctx = canvas.getContext("2d");
    // canvas清屏
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // 重置canvas宽高
    canvas.width = image.width;
    canvas.height = image.height;
    // 将图像绘制到canvas上
    ctx.drawImage(image, 0, 0, image.width, image.height);
    const allCanvas = document.getElementById('allCanvas')
    allCanvas.appendChild(canvas)
    // !!! 注意，image 没有加入到 dom之中
  };
  // 设置src属性，浏览器会自动加载。
  // 记住必须先绑定事件，才能设置src属性，否则会出同步问题。
  image.src = src;
};

// 加载 图像文件(url路径)
function loadImage(src, index, total) {
  // 过滤掉 非 image 类型的文件
  if (!src.type.match(/image.*/)) {
    if (window.console) {
      console.log("选择的文件类型不是图片: ", src.type);
    } else {
      window.confirm("只能选择图片文件");
    }
    return;
  }
  // 创建 FileReader 对象 并调用 render 函数来完成渲染.
  var reader = new FileReader();
  // 绑定load事件自动回调函数
  reader.onload = function (e) {
    // 调用前面的 render 函数
    render(e.target.result);
    check(index, total);
  };
  // 读取文件内容
  reader.readAsDataURL(src);
};

function init() {
  // 获取DOM元素对象
  var target = document.getElementById("drop-target");
  // 阻止 dragover(拖到DOM元素上方) 事件传递
  target.addEventListener("dragover", function (e) { e.preventDefault(); }, true);
  // 拖动并放开鼠标的事件
  target.addEventListener("drop", function (e) {
    // 阻止默认事件，以及事件传播
    e.preventDefault();
    // 调用前面的加载图像 函数，参数为dataTransfer对象的第一个文件
    const checkMsg = document.getElementById('checkMsg');
    const allCanvas = document.getElementById('allCanvas')
    allCanvas.innerHTML = ''
    checkMsg.innerText = '图片中的文字是：';
    const total = e.dataTransfer.files.length;
    window.arr = []
    for (let index = 0; index < total; index++) {
      const file = e.dataTransfer.files[index];
      loadImage(file, index, total);
    }
  }, true);
};

async function check(index, total) {
  let model = window.model
  if (!model) {
    model = await tf.loadLayersModel('http://localhost:8081/trainedModel/model.json');
  }
  ui.logStatus('Starting model training...');
  tf.tidy(() => {
    // const canvas = document.getElementById('myCanvas');
    const allCanvas = document.getElementById('allCanvas')
    const canvas = allCanvas.getElementsByTagName('canvas')[index];
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const datasetBytesBuffer =
      new ArrayBuffer(canvas.width * canvas.height * 4);
    const datasetBytesView = new Float32Array(
      datasetBytesBuffer);
    for (let j = 0; j < imageData.data.length / 4; j++) {
      // All channels hold an equal value since the image is grayscale, so
      // just read the red channel.
      datasetBytesView[j] = imageData.data[j * 4] / 255;
    }
    const a1 = new Float32Array(datasetBytesBuffer)
    const xs = tf.tensor4d(a1, [
      1, canvas.height, canvas.width, 1
    ]);
    const examples = {
      xs
    };
    const output = model.predict(examples.xs);
    const axis = 1;
    const predictions = Array.from(output.argMax(axis).dataSync());
    window.arr[index] = char_set[predictions[0]]
    if (index === total - 1) {
      const checkMsg = document.getElementById('checkMsg');
      checkMsg.innerText += window.arr.join('，');
    }
  });
};

window.onload = () => {
  init()
}

