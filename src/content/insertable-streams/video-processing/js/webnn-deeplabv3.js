'use strict';

/* eslint max-len: ["error", {"code": 120}] */

function sizeOfShape(shape) {
  return shape.reduce((a, b) => {
    return a * b;
  });
}

async function createGPUBuffer(device, size, data = undefined) {
  const sizeInBytes = size * Float32Array.BYTES_PER_ELEMENT;
  const gpuBuffer = device.createBuffer({ size: sizeInBytes, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (data !== undefined) {
    const uploadBuffer = device.createBuffer({ size: sizeInBytes, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC });
    await uploadBuffer.mapAsync(GPUMapMode.WRITE);
    new Float32Array(uploadBuffer.getMappedRange()).set(data);
    uploadBuffer.unmap();
    const uploadEncoder = device.createCommandEncoder();
    uploadEncoder.copyBufferToBuffer(uploadBuffer, 0, gpuBuffer, 0, sizeInBytes);
    device.queue.submit([uploadEncoder.finish()]);
  }
  return gpuBuffer;
}

async function buildConstantByNpy(device, builder, url) {
  const dataTypeMap = new Map([
    ['f2', { type: 'float16', array: Uint16Array }],
    ['f4', { type: 'float32', array: Float32Array }],
    ['f8', { type: 'float64', array: Float64Array }],
    ['i1', { type: 'int8', array: Int8Array }],
    ['i2', { type: 'int16', array: Int16Array }],
    ['i4', { type: 'int32', array: Int32Array }],
    ['i8', { type: 'int64', array: BigInt64Array }],
    ['u1', { type: 'uint8', array: Uint8Array }],
    ['u2', { type: 'uint16', array: Uint16Array }],
    ['u4', { type: 'uint32', array: Uint32Array }],
    ['u8', { type: 'uint64', array: BigUint64Array }],
  ]);
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const npArray = new numpy.Array(new Uint8Array(buffer));
  if (!dataTypeMap.has(npArray.dataType)) {
    throw new Error(`Data type ${npArray.dataType} is not supported.`);
  }
  const dimensions = npArray.shape;
  const type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
  const dataView = new DataView(npArray.data.buffer);
  const littleEndian = npArray.byteOrder === '<';
  for (let i = 0; i < sizeOfShape(dimensions); ++i) {
    typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
      i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
  }
  return builder.constant({ type, dimensions }, { resource: await createGPUBuffer(device, sizeOfShape(dimensions), typedArray) });
}

// DeepLab V3 MobileNet V2 model with 'nchw' input layout
class DeepLabV3MNV2Nchw {
  constructor() {
    this.weightsUrl_ = './models/deeplabv3_1_default_1_nchw/weights/';
    // Shares the same bias files with 'nhwc' layout
    this.biasUrl_ = this.weightsUrl_;
    this.inputOptions = {
      mean: [127.5, 127.5, 127.5],
      std: [127.5, 127.5, 127.5],
      scaledFlag: true,
      inputLayout: 'nchw',
      inputDimensions: [1, 3, 513, 513],
    };
    this.outputDimensions = [1, 1, 513, 513];
    this.device_ = null;
    this.builder_ = null;
    this.graph_ = null;
  }

  async buildConv_(input, nameArray, activation = 'relu6', options = {}) {
    // nameArray: 0: bias name prefix, 1: depthWise Conv2D's bias name suffix, 2: indice of weight name
    const biasPrefix = this.biasUrl_ + nameArray[0];
    const weightsName = `${this.weightsUrl_}const_fold_opt__${nameArray[1]}.npy`;
    const biasName = biasPrefix + '_bias.npy';

    const weights = await buildConstantByNpy(this.device_, this.builder_, weightsName);
    const bias = await buildConstantByNpy(this.device_, this.builder_, biasName);

    options.bias = bias;
    if (activation === 'relu6') {
      // implement `relu6` by `clamp` of  WebNN API
      options.activation = this.builder_.clamp({minValue: 0, maxValue: 6});
    } else if (activation === 'relu') {
      options.activation = this.builder_.relu();
    } else {
      options.activation = undefined;
    }
    return this.builder_.conv2d(input, weights, options);
  }

  async buildLinearBottleneck_(input, nameArray, dwiseOptions, shortcut = true) {
    // nameArray: 0: indice of bias name, 1: indice of conv1x1Relu6's weight name,
    // 2: indice of dwise3x3Relu6's weight name, 3: indice of conv1x1Linear's weight name
    const biasPrefix = 'MobilenetV2_expanded_conv_' + nameArray[0];
    const conv1x1Relu6 = await this.buildConv_(
        input,
        [`${biasPrefix}_expand_Conv2D`, nameArray[1]]);
    const dwise3x3Relu6 = await this.buildConv_(
        conv1x1Relu6,
        [`${biasPrefix}_depthwise_depthwise`, nameArray[2]],
        'relu6',
        dwiseOptions);
    const conv1x1Linear = await this.buildConv_(
        dwise3x3Relu6,
        [`${biasPrefix}_project_Conv2D`, nameArray[3]],
        'none');

    if (shortcut) {
      return this.builder_.add(input, conv1x1Linear);
    }
    return conv1x1Linear;
  }

  async init(device) {
    this.device_ = device;
    const context = navigator.ml.createContext(this.device_);
    this.builder_ = new MLGraphBuilder(context);
    const strides = [2, 2];

    const input = this.builder_.input('input',
        {type: 'float32', dimensions: this.inputOptions.inputDimensions});
    const conv0 = await this.buildConv_(
        input, ['MobilenetV2_Conv_Conv2D', '409'], 'relu6', {strides, padding: [1, 1, 1, 1]});
    const conv1 = await this.buildConv_(
        conv0, ['MobilenetV2_expanded_conv_depthwise_depthwise', '392'], 'relu6',
        {padding: [1, 1, 1, 1], groups: 16});
    const conv2 = await this.buildConv_(
        conv1, ['MobilenetV2_expanded_conv_project_Conv2D', '374'], 'none');
    const bottleneck0 = await this.buildLinearBottleneck_(
        conv2, ['1', '344', '459', '455'], {strides, padding: [1, 1, 1, 1], groups: 48}, false);
    const bottleneck1 = await this.buildLinearBottleneck_(
        bottleneck0, ['2', '405', '419', '447'], {padding: [1, 1, 1, 1], groups: 72});
    const bottleneck2 = await this.buildLinearBottleneck_(
        bottleneck1, ['3', '369', '445', '415'], {strides, padding: [1, 1, 1, 1], groups: 72}, false);
    const bottleneck3 = await this.buildLinearBottleneck_(
        bottleneck2, ['4', '389', '407', '442'], {padding: [1, 1, 1, 1], groups: 96});
    const bottleneck4 = await this.buildLinearBottleneck_(
        bottleneck3, ['5', '411', '461', '468'], {padding: [1, 1, 1, 1], groups: 96});
    const bottleneck5 = await this.buildLinearBottleneck_(
        bottleneck4, ['6', '364', '463', '361'], {padding: [1, 1, 1, 1], groups: 96}, false);
    const bottleneck6 = await this.buildLinearBottleneck_(
        bottleneck5, ['7', '352', '433', '427'], {padding: [2, 2, 2, 2], groups: 192, dilations: [2, 2]});
    const bottleneck7 = await this.buildLinearBottleneck_(
        bottleneck6, ['8', '449', '422', '394'], {padding: [2, 2, 2, 2], groups: 192, dilations: [2, 2]});
    const bottleneck8 = await this.buildLinearBottleneck_(
        bottleneck7, ['9', '462', '384', '452'], {padding: [2, 2, 2, 2], groups: 192, dilations: [2, 2]});
    const bottleneck9 = await this.buildLinearBottleneck_(
        bottleneck8, ['10', '448', '457', '425'], {padding: [2, 2, 2, 2], groups: 192, dilations: [2, 2]}, false);
    const bottleneck10 = await this.buildLinearBottleneck_(
        bottleneck9, ['11', '450', '439', '397'], {padding: [2, 2, 2, 2], groups: 288, dilations: [2, 2]});
    const bottleneck11 = await this.buildLinearBottleneck_(
        bottleneck10, ['12', '377', '465', '372'], {padding: [2, 2, 2, 2], groups: 288, dilations: [2, 2]});
    const bottleneck12 = await this.buildLinearBottleneck_(
        bottleneck11, ['13', '432', '436', '355'], {padding: [2, 2, 2, 2], groups: 288, dilations: [2, 2]}, false);
    const bottleneck13 = await this.buildLinearBottleneck_(
        bottleneck12, ['14', '383', '410', '440'], {padding: [4, 4, 4, 4], groups: 480, dilations: [4, 4]});
    const bottleneck14 = await this.buildLinearBottleneck_(
        bottleneck13, ['15', '424', '426', '420'], {padding: [4, 4, 4, 4], groups: 480, dilations: [4, 4]});
    const bottleneck15 = await this.buildLinearBottleneck_(
        bottleneck14, ['16', '466', '456', '347'], {padding: [4, 4, 4, 4], groups: 480, dilations: [4, 4]}, false);

    const conv3 = await this.buildConv_(bottleneck15, ['aspp0_Conv2D', '399'], 'relu');
    const averagePool2d = this.builder_.averagePool2d(
        bottleneck15, {windowDimensions: [65, 65], layout: 'nchw'});
    const conv4 = await this.buildConv_(averagePool2d, ['image_pooling_Conv2D', '378'], 'relu');
    const resample0 = this.builder_.resample2d(
        conv4, {sizes: [65, 65], mode: 'linear'});
    const concat = this.builder_.concat([resample0, conv3], 1);

    const conv5 = await this.buildConv_(concat, ['concat_projection_Conv2D', '387'], 'relu');
    const conv6 = await this.buildConv_(conv5, ['logits_semantic_Conv2D', '373'], 'none');
    const resample1 = this.builder_.resample2d(
        conv6, {sizes: [65, 65], mode: 'linear'});
    const resample2 = this.builder_.resample2d(
        resample1, {sizes: [513, 513], mode: 'linear'});
    const argmax = this.builder_.reduceArgMax(resample2, {axes: [1], keepDimensions: false});
    this.graph_ = this.builder_.build({'output': argmax});
  }

  async compute(inputGPUBuffer, outputGPUBuffer) {
    this.graph_.compute({'input': {resource: inputGPUBuffer}}, {'output': {resource: outputGPUBuffer}});
  }
}
