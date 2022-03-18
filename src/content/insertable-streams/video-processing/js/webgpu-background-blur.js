/*
 *  Copyright (c) 2022 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

'use strict';

const preprocessWGSL = `
[[block]] struct Tensor {
  values: array<f32>;
};

[[group(0), binding(0)]] var samp : sampler;
[[group(0), binding(1)]] var<storage, write> inputTensor : Tensor;
[[group(0), binding(2)]] var inputTex : texture_2d<f32>;

[[stage(compute), workgroup_size(8, 8)]]
fn main([[builtin(global_invocation_id)]] globalID : vec3<u32>) {
  let dims : vec2<i32> = textureDimensions(inputTex, 0);
  var inputValue = textureSampleLevel(inputTex, samp, vec2<f32>(globalID.xy) / vec2<f32>(dims), 0.0).rgb;
  var normalizedInput = (inputValue * vec3<f32>(255.0, 255.0, 255.0) - vec3<f32>(127.5, 127.5, 127.5)) / vec3<f32>(127.5, 127.5, 127.5);
  inputTensor.values[0u * 513u * 513u + globalID.y * 513u + globalID.x] = normalizedInput.r;
  inputTensor.values[1u * 513u * 513u + globalID.y * 513u + globalID.x] = normalizedInput.g;
  inputTensor.values[2u * 513u * 513u + globalID.y * 513u + globalID.x] = normalizedInput.b;
}
`;

const segWGSL = `
[[block]] struct SegMap {
  labels: array<i32>;
};

[[block]] struct SegMapSize {
  width: u32;
  height: u32;
};

[[group(0), binding(0)]] var samp : sampler;
[[group(0), binding(1)]] var<uniform> segmapSize : SegMapSize;
[[group(0), binding(2)]] var<storage, read> segmap : SegMap;
[[group(0), binding(3)]] var inputTex : texture_2d<f32>;
[[group(0), binding(4)]] var backgroundTex : texture_2d<f32>;
[[group(0), binding(5)]] var outputTex : texture_storage_2d<rgba8unorm, write>;

[[stage(compute), workgroup_size(8, 8)]]
fn main([[builtin(global_invocation_id)]] globalID : vec3<u32>) {
  let dims : vec2<i32> = textureDimensions(inputTex, 0);
  
  var input = textureSampleLevel(inputTex, samp, vec2<f32>(globalID.xy) / vec2<f32>(dims), 0.0).rgb;
  var background = textureSampleLevel(backgroundTex, samp, vec2<f32>(globalID.xy) / vec2<f32>(dims), 0.0).rgb;
  var green : vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
  var segmapX : u32 = u32(floor(f32(globalID.x) / f32(dims.x) * f32(segmapSize.width)));
  var segmapY : u32 = u32(floor(f32(globalID.y) / f32(dims.y) * f32(segmapSize.height)));
  var segmapIndex = segmapX + segmapY * segmapSize.height;
  if (segmap.labels[segmapIndex] == 0) {
    textureStore(outputTex, vec2<i32>(globalID.xy), vec4<f32>(background, 1.0));
  } else {
    textureStore(outputTex, vec2<i32>(globalID.xy), vec4<f32>(input, 1.0));
  }
}
`;

const blurWGSL = `
[[block]] struct Params {
  filterDim : u32;
  blockDim : u32;
};

[[group(0), binding(0)]] var samp : sampler;
[[group(0), binding(1)]] var<uniform> params : Params;
[[group(1), binding(1)]] var inputTex : texture_2d<f32>;
[[group(1), binding(2)]] var outputTex : texture_storage_2d<rgba8unorm, write>;

[[block]] struct Flip {
  value : u32;
};
[[group(1), binding(3)]] var<uniform> flip : Flip;

// This shader blurs the input texture in one direction, depending on whether
// |flip.value| is 0 or 1.
// It does so by running (128 / 4) threads per workgroup to load 128
// texels into 4 rows of shared memory. Each thread loads a
// 4 x 4 block of texels to take advantage of the texture sampling
// hardware.
// Then, each thread computes the blur result by averaging the adjacent texel values
// in shared memory.
// Because we're operating on a subset of the texture, we cannot compute all of the
// results since not all of the neighbors are available in shared memory.
// Specifically, with 128 x 128 tiles, we can only compute and write out
// square blocks of size 128 - (filterSize - 1). We compute the number of blocks
// needed in Javascript and dispatch that amount.

var<workgroup> tile : array<array<vec3<f32>, 128>, 4>;

[[stage(compute), workgroup_size(32, 1, 1)]]
fn main(
  [[builtin(workgroup_id)]] WorkGroupID : vec3<u32>,
  [[builtin(local_invocation_id)]] LocalInvocationID : vec3<u32>
) {
  let filterOffset : u32 = (params.filterDim - 1u) / 2u;
  let dims : vec2<i32> = textureDimensions(inputTex, 0);

  let baseIndex = vec2<i32>(
    WorkGroupID.xy * vec2<u32>(params.blockDim, 4u) +
    LocalInvocationID.xy * vec2<u32>(4u, 1u)
  ) - vec2<i32>(i32(filterOffset), 0);

  for (var r : u32 = 0u; r < 4u; r = r + 1u) {
    for (var c : u32 = 0u; c < 4u; c = c + 1u) {
      var loadIndex = baseIndex + vec2<i32>(i32(c), i32(r));
      if (flip.value != 0u) {
        loadIndex = loadIndex.yx;
      }

      tile[r][4u * LocalInvocationID.x + c] =
        textureSampleLevel(inputTex, samp,
          (vec2<f32>(loadIndex) + vec2<f32>(0.25, 0.25)) / vec2<f32>(dims), 0.0).rgb;
    }
  }

  workgroupBarrier();

  for (var r : u32 = 0u; r < 4u; r = r + 1u) {
    for (var c : u32 = 0u; c < 4u; c = c + 1u) {
      var writeIndex = baseIndex + vec2<i32>(i32(c), i32(r));
      if (flip.value != 0u) {
        writeIndex = writeIndex.yx;
      }

      let center : u32 = 4u * LocalInvocationID.x + c;
      if (center >= filterOffset &&
          center < 128u - filterOffset &&
          all(writeIndex < dims)) {
        var acc : vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        for (var f : u32 = 0u; f < params.filterDim; f = f + 1u) {
          var i : u32 = center + f - filterOffset;
          acc = acc + (1.0 / f32(params.filterDim)) * tile[r][i];
        }
        textureStore(outputTex, writeIndex, vec4<f32>(acc, 1.0));
      }
    }
  }
}
`;

const fullscreenTexturedQuadWGSL = `
[[group(0), binding(0)]] var mySampler : sampler;
[[group(0), binding(1)]] var myTexture : texture_2d<f32>;

struct VertexOutput {
  [[builtin(position)]] Position : vec4<f32>;
  [[location(0)]] fragUV : vec2<f32>;
};

[[stage(vertex)]]
fn vert_main([[builtin(vertex_index)]] VertexIndex : u32) -> VertexOutput {
  var pos = array<vec2<f32>, 6>(
      vec2<f32>( 1.0,  1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0,  1.0));

  var uv = array<vec2<f32>, 6>(
      vec2<f32>(1.0, 0.0),
      vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(0.0, 0.0));

  var output : VertexOutput;
  output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}

[[stage(fragment)]]
fn frag_main([[location(0)]] fragUV : vec2<f32>) -> [[location(0)]] vec4<f32> {
  return textureSample(myTexture, mySampler, fragUV);
}
`;

// Contants from the blur.wgsl shader.
const tileDim = 128;
const batch = [4, 4];

/**
 * Segmentation using WebNN and applies a blur effect using WebGPU.
 * @implements {FrameTransform} in pipeline.js
 */
 class WebGPUBackgroundBlurTransform { // eslint-disable-line no-unused-vars
  constructor() {
    // All fields are initialized in init()
    /** @private {?OffscreenCanvas} canvas used to render video frame */
    this.canvas_ = null;
    /** @private {string} */
    this.debugPath_ = 'debug.pipeline.frameTransform_';

    this.context_ = null;
    this.device_ = null;
    this.adapter_ = null;

    this.settings_ = {
      filterSize: 15,
      iterations: 2,
    };
    this.segmentationSize_ = {width: 513, height: 513};
    this.segmapBuffer_ = null;

    this.deeplab_ = null;

    this.blurBackgroundCheckbox_ = (/** @type {!HTMLInputElement} */ (
      document.getElementById('blurBackground')));
  }

  /** @override */
  async init() {
    console.log('[WebGPUBackgroundBlurTransform] Initializing WebGPU.');
    if (!navigator.gpu) {
      alert(
        'Failed to detect WebGPU. Check that WebGPU is supported ' +
        'by your browser and hardware.');
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    this.adapter_ = adapter;
    const device = await adapter.requestDevice();
    if (!device) {
      throw new Error('Failed to create GPUDevice.');
    }
    this.device_ = device;
    const canvas = new OffscreenCanvas(1, 1);
    this.canvas_ = canvas;
    const context = canvas.getContext('webgpu');
    this.context_ = context;

    const segPipeline = device.createComputePipeline({
      compute: {
        module: device.createShaderModule({
          code: segWGSL,
        }),
        entryPoint: 'main',
      },
    });
    this.segPipeline_ = segPipeline;

    const preprocessPipeline = device.createComputePipeline({
      compute: {
        module: device.createShaderModule({
          code: preprocessWGSL,
        }),
        entryPoint: 'main',
      },
    });
    this.preprocessPipeline_ = preprocessPipeline;

    const blurPipeline = device.createComputePipeline({
      compute: {
        module: device.createShaderModule({
          code: blurWGSL,
        }),
        entryPoint: 'main',
      },
    });
    this.blurPipeline_ = blurPipeline;

    const presentationFormat = context.getPreferredFormat(adapter);
    const fullscreenQuadPipeline = device.createRenderPipeline({
      vertex: {
        module: device.createShaderModule({
          code: fullscreenTexturedQuadWGSL,
        }),
        entryPoint: 'vert_main',
      },
      fragment: {
        module: device.createShaderModule({
          code: fullscreenTexturedQuadWGSL,
        }),
        entryPoint: 'frag_main',
        targets: [
          {
            format: presentationFormat,
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
    this.fullscreenQuadPipeline_ = fullscreenQuadPipeline;

    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.sampler_ = sampler;

    const cubeTexture = device.createTexture({
      size: [this.segmentationSize_.width, this.segmentationSize_.height, 1],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.cubeTexture_ = cubeTexture;

    const textures = [0, 1, 2].map(() => {
      return device.createTexture({
        size: {
          width: this.segmentationSize_.width,
          height: this.segmentationSize_.height,
        },
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING,
      });
    });
    this.textures_ = textures;

    const buffer0 = (() => {
      const buffer = device.createBuffer({
        size: 4,
        mappedAtCreation: true,
        usage: GPUBufferUsage.UNIFORM,
      });
      new Uint32Array(buffer.getMappedRange())[0] = 0;
      buffer.unmap();
      return buffer;
    })();
    this.buffer0_ = buffer0;

    const buffer1 = (() => {
      const buffer = device.createBuffer({
        size: 4,
        mappedAtCreation: true,
        usage: GPUBufferUsage.UNIFORM,
      });
      new Uint32Array(buffer.getMappedRange())[0] = 1;
      buffer.unmap();
      return buffer;
    })();
    this.buffer1_ = buffer1;

    const segmapSizeBuffer = (() => {
      const buffer = device.createBuffer({
        size: 2 * Uint32Array.BYTES_PER_ELEMENT,
        mappedAtCreation: true,
        usage: GPUBufferUsage.UNIFORM,
      });
      new Uint32Array(buffer.getMappedRange()).set([this.segmentationSize_.width, this.segmentationSize_.height]);
      buffer.unmap();
      return buffer;
    })();
    this.segmapSizeBuffer_ = segmapSizeBuffer;

    const inputTensorBuffer = (() => {
      const buffer = device.createBuffer({
        size: 3 * this.segmentationSize_.width * this.segmentationSize_.height * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      });
      return buffer;
    })();
    this.inputTensorBuffer_ = inputTensorBuffer;

    const segmapBuffer = (() => {
      const buffer = device.createBuffer({
        size: this.segmentationSize_.width * this.segmentationSize_.height * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
      });
      return buffer;
    })();
    this.segmapBuffer_ = segmapBuffer;

    const blurParamsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
    });

    const computeConstants = device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: sampler,
        },
        {
          binding: 1,
          resource: {
            buffer: blurParamsBuffer,
          },
        },
      ],
    });
    this.computeConstants_ = computeConstants;

    const computeBindGroup1 = device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 1,
          resource: textures[0].createView(),
        },
        {
          binding: 2,
          resource: textures[1].createView(),
        },
        {
          binding: 3,
          resource: {
            buffer: buffer1,
          },
        },
      ],
    });
    this.computeBindGroup1_ = computeBindGroup1;

    const computeBindGroup2 = device.createBindGroup({
      layout: blurPipeline.getBindGroupLayout(1),
      entries: [
        {
          binding: 1,
          resource: textures[1].createView(),
        },
        {
          binding: 2,
          resource: textures[0].createView(),
        },
        {
          binding: 3,
          resource: {
            buffer: buffer0,
          },
        },
      ],
    });
    this.computeBindGroup2_ = computeBindGroup2;

    const settings = this.settings_;

    const blockDim = tileDim - (settings.filterSize - 1);
    device.queue.writeBuffer(
      blurParamsBuffer,
      0,
      new Uint32Array([settings.filterSize, blockDim])
    );

    this.deeplab_ = new DeepLabV3MNV2Nchw()
    await this.deeplab_.init(this.device_);

    console.log(
        '[WebGPUBackgroundBlurTransform] WebGPU initialized.', `${this.debugPath_}.canvas_ =`,
        this.canvas_, `${this.debugPath_}.device_ =`, this.device_);
  }

  /** @override */
  async transform(frame, controller) {
    const device = this.device_;
    const canvas = this.canvas_;
    if (!device || !canvas || !this.deeplab_) {
      frame.close();
      return;
    }

    const isBlurBackground = this.blurBackgroundCheckbox_.checked ? true : false;

    // Set output size to input size
    if (canvas.width !== frame.displayWidth || canvas.height !== frame.displayHeight) {
      canvas.width = frame.displayWidth;
      canvas.height = frame.displayHeight;
      const devicePixelRatio = window.devicePixelRatio || 1;
      const presentationSize = [
        canvas.width * devicePixelRatio,
        canvas.height * devicePixelRatio,
      ];
      const presentationFormat = this.context_.getPreferredFormat(this.adapter_);
      this.context_.configure({
          device: this.device_,
          format: presentationFormat,
          size: presentationSize,
        });
    }
    const width = this.segmentationSize_.width;
    const height = this.segmentationSize_.height;
    const videoBitmap = await createImageBitmap(frame, {resizeWidth: width, resizeHeight: height});
    device.queue.copyExternalImageToTexture(
      { source: videoBitmap },
      { texture: this.cubeTexture_ },
      [ width, height]
    );
    videoBitmap.close();
    const externalResource = this.cubeTexture_.createView();

    const computeSegBindGroup = device.createBindGroup({
      layout: this.segPipeline_.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.sampler_,
        },
        {
          binding: 1,
          resource: {
            buffer: this.segmapSizeBuffer_,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.segmapBuffer_,
          },
        },
        {
          binding: 3,
          resource: externalResource,
        },
        {
          binding: 4,
          resource: this.textures_[0].createView(),
        },
        {
          binding: 5,
          resource: this.textures_[2].createView(),
        },
      ],
    });

    const preprocessBindGroup = device.createBindGroup({
      layout: this.preprocessPipeline_.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.sampler_,
        },
        {
          binding: 1,
          resource: {
            buffer: this.inputTensorBuffer_,
          },
        },
        {
          binding: 2,
          resource: externalResource,
        },
      ],
    });

    const computeBindGroup0 = device.createBindGroup({
      layout: this.blurPipeline_.getBindGroupLayout(1),
      entries: [
        {
          binding: 1,
          resource: externalResource,
        },
        {
          binding: 2,
          resource: this.textures_[0].createView(),
        },
        {
          binding: 3,
          resource: {
            buffer: this.buffer0_,
          },
        },
      ],
    });

    const commandEncoder = device.createCommandEncoder();

    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(this.blurPipeline_);
    computePass.setBindGroup(0, this.computeConstants_);

    const blockDim = tileDim - (this.settings_.filterSize - 1);
    computePass.setBindGroup(1, computeBindGroup0);
    computePass.dispatch(
      Math.ceil(width / blockDim),
      Math.ceil(height / batch[1])
    );

    computePass.setBindGroup(1, this.computeBindGroup1_);
    computePass.dispatch(
      Math.ceil(width / blockDim),
      Math.ceil(height / batch[1])
    );

    for (let i = 0; i < this.settings_.iterations - 1; ++i) {
      computePass.setBindGroup(1, this.computeBindGroup2_);
      computePass.dispatch(
        Math.ceil(width / blockDim),
        Math.ceil(height / batch[1])
      );

      computePass.setBindGroup(1, this.computeBindGroup1_);
      computePass.dispatch(
        Math.ceil(height / blockDim),
        Math.ceil(width / batch[1])
      );
    }

    if (isBlurBackground) {
      computePass.setPipeline(this.preprocessPipeline_);
      computePass.setBindGroup(0, preprocessBindGroup);
      computePass.dispatch(
        Math.ceil(width / 8),
        Math.ceil(height / 8)
      );
      this.deeplab_.compute(this.inputTensorBuffer_, this.segmapBuffer_);
      computePass.setPipeline(this.segPipeline_);
      computePass.setBindGroup(0, computeSegBindGroup);
      computePass.dispatch(
        Math.ceil(width / 8),
        Math.ceil(height / 8)
      );
    }

    computePass.endPass();

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context_.getCurrentTexture().createView(),
          loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          storeOp: 'store',
        },
      ],
    });

    const showResultBindGroup = device.createBindGroup({
      layout: this.fullscreenQuadPipeline_.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.sampler_,
        },
        {
          binding: 1,
          resource: isBlurBackground ? this.textures_[2].createView() : this.textures_[0].createView(),
        },
      ],
    });

    passEncoder.setPipeline(this.fullscreenQuadPipeline_);
    passEncoder.setBindGroup(0, showResultBindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.endPass();
    device.queue.submit([commandEncoder.finish()]);

    await device.queue.onSubmittedWorkDone();

    // Create a video frame from canvas and enqueue it to controller
    // alpha: 'discard' is needed in order to send frames to a PeerConnection.
    frame.close();
    controller.enqueue(new VideoFrame(this.canvas_, {timestamp: frame.timestamp, alpha: 'discard'}));
  }

  /** @override */
  destroy() {
    if (this.device_) {
      console.log('[WebGPUBackgroundBlurTransform] Destory WebGPU device.');
      this.device_.destroy();
      this.device_ = null;
    }
    this.deeplab_ = null;
  }
}
