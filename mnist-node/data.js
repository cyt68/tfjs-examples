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

const tf = require('@tensorflow/tfjs');
const assert = require('assert');
const fs = require('fs');
const https = require('https');
const util = require('util');
const zlib = require('zlib');
const path = require('path');
const char_set = require('./char_set.json');
const { createCanvas, loadImage } = require('canvas')

const readFile = util.promisify(fs.readFile);

// MNIST data constants:
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte';
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte';
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

// Downloads a test file only once and returns the buffer for the file.
async function fetchOnceAndSaveToDiskWithBuffer(filename) {
  return new Promise(resolve => {
    const url = `${BASE_URL}${filename}.gz`;
    if (fs.existsSync(filename)) {
      resolve(readFile(filename));
      return;
    }
    const file = fs.createWriteStream(filename);
    console.log(`  * Downloading from: ${url}`);
    https.get(url, (response) => {
      const unzip = zlib.createGunzip();
      response.pipe(unzip).pipe(file);
      unzip.on('end', () => {
        resolve(readFile(filename));
      });
    });
  });
}

function loadHeaderValues(buffer, headerLength) {
  const headerValues = [];
  for (let i = 0; i < headerLength / 4; i++) {
    // Header data is stored in-order (aka big-endian)
    headerValues[i] = buffer.readUInt32BE(i * 4);
  }
  return headerValues;
}

async function loadImages(filename) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const headerBytes = IMAGE_HEADER_BYTES;
  const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], IMAGE_HEADER_MAGIC_NUM);
  assert.equal(headerValues[2], IMAGE_HEIGHT);
  assert.equal(headerValues[3], IMAGE_WIDTH);

  const images = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Float32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      // Normalize the pixel values into the 0-1 interval, from
      // the original 0-255 interval.
      array[i] = buffer.readUInt8(index++) / 255;
    }
    images.push(array);
  }

  assert.equal(images.length, headerValues[1]);
  return images;
}

function notDSStore(name) {
  return name !== '.DS_Store'
}

function emptyDir(path) {
  const files = fs.readdirSync(path);
  files.forEach(file => {
    const filePath = `${path}/${file}`;
    const stats = fs.statSync(filePath);
    if (stats.isDirectory()) {
      emptyDir(filePath);
    } else {
      fs.unlinkSync(filePath);
      console.log(`删除${file}文件成功`);
    }
  });
}

async function loadPngs(url) {
  url.includes('train') && emptyDir(path.join(__dirname, `/111`))
  const images = [];
  const sizes = [];
  const labels = [];
  const files = fs.readdirSync(url, { withFileTypes: true });
  for (let i = 0; i < files.length; i++) {
    const el = files[i];
    if (el.isDirectory() && notDSStore(el.name)) {
      if (parseInt(el.name) > 10) {
        break
      }
      let files = fs.readdirSync(`${url}/${el.name}`);
      files = files.filter(el => notDSStore(el))
      files.sort((a, b) => {
        return a.split('.')[0] - b.split('.')[0]
      })
      for (let index = 0; index < files.length; index++) {
        const name = files[index];
        const img = await loadImage(`${url}/${el.name}/${name}`);
        const canvas = createCanvas(img.width, img.height)
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        // if (name.includes('1328') || name.includes('2585')) {
        //   const data = canvas.toBuffer()
        //   fs.writeFileSync(path.join(__dirname, `/111/1-${name}`), data)
        // }
        const datasetBytesBuffer = new ArrayBuffer(img.width * img.height * 4);
        const datasetBytesView = new Float32Array(datasetBytesBuffer);
        for (let j = 0; j < imageData.data.length / 4; j++) {
          // All channels hold an equal value since the image is grayscale, so
          // just read the red channel.
          datasetBytesView[j] = imageData.data[j * 4] / 255;
        }
        const data = resizePng(new Float32Array(datasetBytesBuffer), img, name, parseInt(el.name))
        if (name.includes('65998')) {
          data
        }
        images.push(data)
        sizes.push([img.height, img.width])

        const array = new Int32Array(1);
        array[0] = parseInt(el.name)
        labels.push(array)
      }
    }
  }
  // const disorderObj = {
  //   images: [],
  //   labels: [],
  //   sizes: []
  // }
  // const disorderArr = Array(images.length).fill().map((_, i) => i).sort(() => (Math.random() - 0.5))
  // for (let index = 0; index < disorderArr.length; index++) {
  //   const num = disorderArr[index];
  //   disorderObj.images[index] = images[num]
  //   disorderObj.labels[index] = labels[num]
  //   disorderObj.sizes[index] = sizes[num]
  // }
  // return disorderObj
  return { images, labels, sizes }
}

function fillPicture(dataBuffer, img) {
  const imgHeight = img.height;
  const imgWidth = img.width;
  if (imgHeight < IMAGE_HEIGHT) {
    const diffHeight = IMAGE_HEIGHT - imgHeight
    const halfAddNum = imgWidth * Math.ceil(diffHeight / 2)
    const LeftAddNum = imgWidth * (diffHeight - Math.ceil(diffHeight / 2))
    const imgData = new Float32Array(IMAGE_HEIGHT * imgWidth);
    for (let index = 0; index < halfAddNum; index++) {
      imgData[index] = 1
    }
    for (let index = halfAddNum; index < imgHeight * imgWidth + halfAddNum; index++) {
      imgData[index] = dataBuffer[index - halfAddNum]
    }
    for (let index = imgHeight * imgWidth + halfAddNum; index < imgHeight * imgWidth + halfAddNum + LeftAddNum; index++) {
      imgData[index] = 1
    }
    return {
      data: imgData,
      height: IMAGE_HEIGHT,
      width: imgWidth
    }
  }
  return {
    data: dataBuffer,
    height: imgHeight,
    width: imgWidth
  };
}


function resizePng(dataBuffer, img, name, elName) {
  const { data, height, width } = fillPicture(dataBuffer, img)
  const tmp = tf.tensor4d(data, [
    1, height, width, 1
  ]);

  const res = tmp.resizeBilinear([IMAGE_HEIGHT, IMAGE_WIDTH]).bufferSync().values
  const canvas = createCanvas(IMAGE_HEIGHT, IMAGE_WIDTH)
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);
  for (let i = 0; i < IMAGE_HEIGHT * IMAGE_WIDTH; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = res[i] * 255;
    imageData.data[j + 1] = res[i] * 255;
    imageData.data[j + 2] = res[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  const canvasData = canvas.toBuffer()
  fs.writeFileSync(path.join(__dirname, `/111/${name}`), canvasData)
  return res
}

async function loadLabels(filename) {
  const buffer = await fetchOnceAndSaveToDiskWithBuffer(filename);

  const headerBytes = LABEL_HEADER_BYTES;
  const recordBytes = LABEL_RECORD_BYTE;

  const headerValues = loadHeaderValues(buffer, headerBytes);
  assert.equal(headerValues[0], LABEL_HEADER_MAGIC_NUM);

  const labels = [];
  let index = headerBytes;
  while (index < buffer.byteLength) {
    const array = new Int32Array(recordBytes);
    for (let i = 0; i < recordBytes; i++) {
      array[i] = buffer.readUInt8(index++);
    }
    labels.push(array);
  }

  assert.equal(labels.length, headerValues[1]);
  return labels;
}

/** Helper class to handle loading training and test data. */
class MnistDataset {
  constructor() {
    this.dataset = null;
    this.trainSize = 0;
    this.testSize = 0;
    this.trainBatchIndex = 0;
    this.testBatchIndex = 0;
  }

  /** Loads training and test data. */
  async loadData() {
    // this.dataset = await Promise.all([
    //   loadImages(TRAIN_IMAGES_FILE), loadLabels(TRAIN_LABELS_FILE),
    //   loadImages(TEST_IMAGES_FILE), loadLabels(TEST_LABELS_FILE)
    // ]);
    const trainData = await loadPngs(path.join(__dirname, `/data/train`))
    const testData = await loadPngs(path.join(__dirname, `/data/test`))
    this.dataset = await Promise.all([
      trainData.images, trainData.labels,
      testData.images, testData.labels, trainData.sizes,
    ]);
    this.trainSize = this.dataset[0].length;
    this.testSize = this.dataset[2].length;
  }

  getTrainData() {
    return this.getData_(true);
  }

  getTestData() {
    return this.getData_(false);
  }

  getData_(isTrainingData) {
    let imagesIndex;
    let labelsIndex;
    if (isTrainingData) {
      imagesIndex = 0;
      labelsIndex = 1;
    } else {
      imagesIndex = 2;
      labelsIndex = 3;
    }
    const size = this.dataset[imagesIndex].length;
    tf.util.assert(
      this.dataset[labelsIndex].length === size,
      `Mismatch in the number of images (${size}) and ` +
      `the number of labels (${this.dataset[labelsIndex].length})`);

    // Only create one big array to hold batch of images.
    const imagesShape = [size, IMAGE_HEIGHT, IMAGE_WIDTH, 1];
    const images = new Float32Array(tf.util.sizeFromShape(imagesShape));
    const labels = new Int32Array(tf.util.sizeFromShape([size, 1]));

    let imageOffset = 0;
    let labelOffset = 0;
    for (let i = 0; i < size; ++i) {
      images.set(this.dataset[imagesIndex][i], imageOffset);
      labels.set(this.dataset[labelsIndex][i], labelOffset);
      imageOffset += IMAGE_FLAT_SIZE;
      labelOffset += 1;
    }

    return {
      images: tf.tensor4d(images, imagesShape),
      labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat()
    };
  }
}

module.exports = new MnistDataset();
