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

import * as tfvis from '@tensorflow/tfjs-vis';
import char_set from './char_set.json';

const statusElement = document.getElementById('status');
const messageElement = document.getElementById('message');
const imagesElement = document.getElementById('images');

export function logStatus(message) {
  statusElement.innerText = message;
}

export function trainingLog(message) {
  messageElement.innerText = `${message}\n`;
  console.log(message);
}

export function showTestResults(batch, predictions, labels) {
  const testExamples = batch.xs.shape[0];
  imagesElement.innerHTML = '';
  for (let i = 0; i < testExamples; i++) {
    const image = batch.xs.slice([i, 0], [1, batch.xs.shape[1]]);

    const div = document.createElement('div');
    div.className = 'pred-container';

    const canvas = document.createElement('canvas');
    canvas.className = 'prediction-canvas';
    draw(image.flatten(), canvas);

    const pred = document.createElement('div');

    const prediction = predictions[i];
    const label = labels[i];
    const correct = prediction === label;

    pred.className = `pred ${(correct ? 'pred-correct' : 'pred-incorrect')}`;
    pred.innerText = `pred: ${char_set[prediction]}`;

    div.appendChild(pred);
    div.appendChild(canvas);

    imagesElement.appendChild(div);
  }
}

const lossLabelElement = document.getElementById('loss-label');
const accuracyLabelElement = document.getElementById('accuracy-label');
const lossValues = [[], []];
export function plotLoss(batch, loss, set) {
  const series = set === 'train' ? 0 : 1;
  lossValues[series].push({ x: batch, y: loss });
  const lossContainer = document.getElementById('loss-canvas');
  tfvis.render.linechart(
    lossContainer, { values: lossValues, series: ['train', 'validation'] }, {
    xLabel: 'Batch #',
    yLabel: 'Loss',
    width: 400,
    height: 300,
  });
  lossLabelElement.innerText = `last loss: ${loss.toFixed(3)}`;
}

const accuracyValues = [[], []];
export function plotAccuracy(batch, accuracy, set) {
  const accuracyContainer = document.getElementById('accuracy-canvas');
  const series = set === 'train' ? 0 : 1;
  accuracyValues[series].push({ x: batch, y: accuracy });
  tfvis.render.linechart(
    accuracyContainer,
    { values: accuracyValues, series: ['train', 'validation'] }, {
    xLabel: 'Batch #',
    yLabel: 'Accuracy',
    width: 400,
    height: 300,
  });
  accuracyLabelElement.innerText =
    `last accuracy: ${(accuracy * 100).toFixed(1)}%`;
}

export function draw(image, canvas) {
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function getModelTypeId() {
  return document.getElementById('model-type').value;
}

export function getTrainEpochs() {
  return Number.parseInt(document.getElementById('train-epochs').value);
}

export function setTrainButtonCallback(callback) {
  const trainButton = document.getElementById('train');
  const modelType = document.getElementById('model-type');
  trainButton.addEventListener('click', () => {
    trainButton.setAttribute('disabled', true);
    modelType.setAttribute('disabled', true);
    callback();
  });
}

export function setTestCallback(callback) {
  const btn = document.getElementById('test');
  btn.addEventListener('click', () => {
    // btn.setAttribute('disabled', true);
    callback();
  });
}

export function setCheckCallback(callback) {
  const btn = document.getElementById('check');
  btn.addEventListener('click', () => {
    // btn.setAttribute('disabled', true);
    callback();
  });
}

var MAX_HEIGHT = 100;
// 渲染
function render(src) {
  // 创建一个 Image 对象
  var image = new Image();
  // 绑定 load 事件处理器，加载完成后执行
  image.onload = function () {
    // 获取 canvas DOM 对象
    var canvas = document.getElementById("myCanvas");
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
    // !!! 注意，image 没有加入到 dom之中
  };
  // 设置src属性，浏览器会自动加载。
  // 记住必须先绑定事件，才能设置src属性，否则会出同步问题。
  image.src = src;
};

// 加载 图像文件(url路径)
function loadImage(src) {
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
    loadImage(e.dataTransfer.files[0]);
  }, true);
};

window.onload = () => {
  init()
}
