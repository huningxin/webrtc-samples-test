/*
 *  Copyright (c) 2022 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

'use strict';

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
 * Applies a blur effect using WebGPU.
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

    this.blurSettings_ = {
      filterSize: 10,
      iterations: 2,
    };
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

    const settings = this.blurSettings_;

    const blockDim = tileDim - (settings.filterSize - 1);
    device.queue.writeBuffer(
      blurParamsBuffer,
      0,
      new Uint32Array([settings.filterSize, blockDim])
    );

    console.log(
        '[WebGPUBackgroundBlurTransform] WebGPU initialized.', `${this.debugPath_}.canvas_ =`,
        this.canvas_, `${this.debugPath_}.device_ =`, this.device_);
  }

  initResources_(frameWidth, frameHeight) {
    const device = this.device_;
    const cubeTexture = device.createTexture({
      size: [frameWidth, frameHeight, 1],
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
          width: frameWidth,
          height: frameHeight,
        },
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING,
      });
    });
    this.textures_ = textures;
  }

  /** @override */
  async transform(frame, controller) {
    const device = this.device_;
    const canvas = this.canvas_;
    if (!device || !canvas) {
      frame.close();
      return;
    }

    // Set output size to input size
    const frameWidth = frame.displayWidth;
    const frameHeight = frame.displayHeight;
    if (canvas.width !== frameWidth || canvas.height !== frameHeight) {
      canvas.width = frameWidth;
      canvas.height = frameHeight;
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
      this.initResources_(frameWidth, frameHeight);
    }
    const videoBitmap = await createImageBitmap(frame);
    device.queue.copyExternalImageToTexture(
      { source: videoBitmap },
      { texture: this.cubeTexture_ },
      [frameWidth, frameHeight]
    );
    videoBitmap.close();
    const externalResource = this.cubeTexture_.createView();

    const blurBindGroup0 = device.createBindGroup({
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

    const blurBindGroup1 = device.createBindGroup({
      layout: this.blurPipeline_.getBindGroupLayout(1),
      entries: [
        {
          binding: 1,
          resource: this.textures_[0].createView(),
        },
        {
          binding: 2,
          resource: this.textures_[1].createView(),
        },
        {
          binding: 3,
          resource: {
            buffer: this.buffer1_,
          },
        },
      ],
    });

    const blurBindGroup2 = device.createBindGroup({
      layout: this.blurPipeline_.getBindGroupLayout(1),
      entries: [
        {
          binding: 1,
          resource: this.textures_[1].createView(),
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

    const blockDim = tileDim - (this.blurSettings_.filterSize - 1);
    computePass.setBindGroup(1, blurBindGroup0);
    computePass.dispatch(
      Math.ceil(frameWidth / blockDim),
      Math.ceil(frameHeight / batch[1])
    );

    computePass.setBindGroup(1, blurBindGroup1);
    computePass.dispatch(
      Math.ceil(frameHeight / blockDim),
      Math.ceil(frameWidth / batch[1])
    );

    for (let i = 0; i < this.blurSettings_.iterations - 1; ++i) {
      computePass.setBindGroup(1, blurBindGroup2);
      computePass.dispatch(
        Math.ceil(frameWidth / blockDim),
        Math.ceil(frameHeight / batch[1])
      );

      computePass.setBindGroup(1, blurBindGroup1);
      computePass.dispatch(
        Math.ceil(frameHeight / blockDim),
        Math.ceil(frameWidth / batch[1])
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
          resource: this.textures_[0].createView(),
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
