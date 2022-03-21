/*
 *  Copyright (c) 2022 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree.
 */

'use strict';

const customBackendName = 'custom-webgl';

/**
 * Applies a blur effect using WebGL.
 * @implements {FrameTransform} in pipeline.js
 */
class WebGLBackgroundBlurTransform { // eslint-disable-line no-unused-vars
  constructor() {
    // All fields are initialized in init()
    /** @private {?OffscreenCanvas} canvas used to create the WebGL context */
    this.canvas_ = null;
    /** @private {?WebGL2RenderingContext} */
    this.gl_ = null;
    /** @private {string} */
    this.debugPath_ = 'debug.pipeline.frameTransform_';

    // Input texture
    this.inputTexture_ = null;

    // Resize program
    this.resizeProgram_ = null;
    this.resizeProgramInputSampler_ = null;

    // Blur program
    this.blurProgram_ = null;
    this.blurProgramInputSampler_ = null;
    this.blurProgramSegmentationSampler_ = null;
    this.texelSizeLocation_ = null;
    this.blurBackgroundLocation_ = null;
  }

  /** @override */
  async init() {
    console.log('[WebGLBackgroundBlur] Initializing WebGL.');
    this.canvas_ = new OffscreenCanvas(1, 1);
    const gl = /** @type {?WebGL2RenderingContext} */ (
      this.canvas_.getContext('webgl2'));
    if (!gl) {
      alert(
          'Failed to create WebGL2 context. Check that WebGL2 is supported ' +
          'by your browser and hardware.');
      return;
    }
    this.gl_ = gl;
    const vertexShaderSrc = `#version 300 es
      precision highp float;
      in vec2 a_position;
      in vec2 a_texCoord;
      out vec2 v_texCoord;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        v_texCoord = a_texCoord;
      }`;
    const resizeFragmentShaderSrc = `#version 300 es
      precision highp float;
      uniform sampler2D u_inputFrame;
      in vec2 v_texCoord;
      out vec4 outColor;
      void main() {
        outColor = texture(u_inputFrame, vec2(v_texCoord[0], 1.0 - v_texCoord[1]));
      }`;
    this.resizeProgram_ = this.createProgram_(vertexShaderSrc, resizeFragmentShaderSrc);
    this.resizeProgramInputSampler_ = gl.getUniformLocation(this.resizeProgram_, 'u_inputFrame');
    const blurFragmentShaderSrc = `#version 300 es
      precision highp float;
      uniform sampler2D u_inputFrame;
      uniform vec2 u_texelSize;
      in vec2 v_texCoord;
      out vec4 outColor;
      const float offset[5] = float[](0.0, 1.0, 2.0, 3.0, 4.0);
      const float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

      void main() {
        vec4 centerColor = texture(u_inputFrame, v_texCoord);
        vec4 frameColor = centerColor * weight[0];
        for (int i = 1; i < 5; i++) {
          vec2 offset = vec2(offset[i]) * u_texelSize;
          vec2 texCoord = v_texCoord + offset;
          frameColor += texture(u_inputFrame, texCoord) * weight[i];
          texCoord = v_texCoord - offset;
          frameColor += texture(u_inputFrame, texCoord) * weight[i];
        }
        outColor = vec4(frameColor.rgb + (1.0 - frameColor.a) * centerColor.rgb, 1.0);
      }`;
    this.blurProgram_ = this.createProgram_(vertexShaderSrc, blurFragmentShaderSrc);
    this.blurProgramInputSampler_ = gl.getUniformLocation(this.blurProgram_, 'u_inputFrame');
    this.texelSizeLocation_ = gl.getUniformLocation(this.blurProgram_, 'u_texelSize');

    console.log(
        '[WebGLBackgroundBlur] WebGL initialized.', `${this.debugPath_}.canvas_ =`,
        this.canvas_, `${this.debugPath_}.gl_ =`, this.gl_);
  }

  initTextures_(frameWidth, frameHeight) {
    // Initialize input texture
    this.inputTexture_ = this.createTexture_();
    this.texture1_ = this.createTexture_(frameWidth, frameHeight);
    this.frameBuffer1_ = this.createFramebuffer_(this.texture1_);
    this.texture2_ = this.createTexture_(frameWidth, frameHeight);
    this.frameBuffer2_ = this.createFramebuffer_(this.texture2_);
  }

  /**
   * Creates and compiles a WebGLShader from the provided source code.
   * @param {number} type either VERTEX_SHADER or FRAGMENT_SHADER
   * @param {string} shaderSrc
   * @return {!WebGLShader}
   * @private
   */
  loadShader_(type, shaderSrc) {
    const gl = this.gl_;
    const shader = gl.createShader(type);
    // Load the shader source
    gl.shaderSource(shader, shaderSrc);
    // Compile the shader
    gl.compileShader(shader);
    // Check the compile status
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const infoLog = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Error compiling shader:\n${infoLog}`);
    }
    return shader;
  }

  /**
   * Sets a floating point shader attribute to the values in arr.
   * @param {WebGLProgram} program the WebGL program to set attributes
   * @param {string} attrName the name of the shader attribute to set
   * @param {number} vsize the number of components of the shader attribute's
   *   type
   * @param {!Array<number>} arr the values to set
   * @private
   */
  attributeSetFloats_(program, attrName, vsize, arr) {
    const gl = this.gl_;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
    const attr = gl.getAttribLocation(program, attrName);
    gl.enableVertexAttribArray(attr);
    gl.vertexAttribPointer(attr, vsize, gl.FLOAT, false, 0, 0);
  }

  createTexture_(width = 0, height = 0, internalformat = this.gl_.RGBA8, minFilter = this.gl_.NEAREST, magFilter = this.gl_.NEAREST) {
    const gl = this.gl_;
    const texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
    if (width !== 0 && height !== 0) {
      gl.texStorage2D(gl.TEXTURE_2D, 1, internalformat, width, height);
    }
    return texture;
  }

  createFramebuffer_(texture) {
    const gl = this.gl_;
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0
    );
    return frameBuffer;
  }

  createProgram_(vertexShaderSrc, fragmentShaderSrc) {
    const gl = this.gl_;
    const vertexShader = this.loadShader_(gl.VERTEX_SHADER, vertexShaderSrc);
    const fragmentShader = this.loadShader_(gl.FRAGMENT_SHADER, fragmentShaderSrc);
    if (!vertexShader || !fragmentShader) {
      throw new Error('Failed to load shader');
    }
    // Create the program object
    const programObject = gl.createProgram();
    gl.attachShader(programObject, vertexShader);
    gl.attachShader(programObject, fragmentShader);
    // Link the program
    gl.linkProgram(programObject);
    // Check the link status
    const linked = gl.getProgramParameter(programObject, gl.LINK_STATUS);
    if (!linked) {
      const infoLog = gl.getProgramInfoLog(programObject);
      gl.deleteProgram(programObject);
      throw new Error(`Error linking program:\n${infoLog}`);
    }
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    this.attributeSetFloats_(programObject, 'a_position', 2, [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
    this.attributeSetFloats_(programObject, 'a_texCoord', 2, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    return programObject;
  }

  /** @override */
  async transform(frame, controller) {
    const gl = this.gl_;
    if (!gl || !this.canvas_) {
      frame.close();
      return;
    }
    // Set output size to input size
    const frameWidth = frame.displayWidth;
    const frameHeight = frame.displayHeight;
    if (this.canvas_.width !== frameWidth || this.canvas_.height !== frameHeight) {
      this.canvas_.width = frameWidth;
      this.canvas_.height = frameHeight;
      this.initTextures_(frameWidth, frameHeight);
    }

    const videoBitmap = await createImageBitmap(frame);
    // Upload frame to input texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.inputTexture_);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoBitmap);
    videoBitmap.close();

    // Blur
    const texelWidth = 1 / frameWidth;
    const texelHeight = 1 / frameHeight;
    gl.viewport(0, 0, frameWidth, frameHeight);
    gl.scissor(0, 0, frameWidth, frameHeight);
    gl.useProgram(this.blurProgram_);
    gl.uniform1i(this.blurProgramInputSampler_, 0);
    for (let i = 0; i < 3; i++) {
      gl.uniform2f(this.texelSizeLocation_, 0, texelHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer1_);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      gl.activeTexture(gl.TEXTURE1)
      gl.bindTexture(gl.TEXTURE_2D, this.texture1_);
      gl.uniform1i(this.blurProgramInputSampler_, 1);

      gl.uniform2f(this.texelSizeLocation_, texelWidth, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer2_);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.bindTexture(gl.TEXTURE_2D, this.texture2_);
    }

    gl.viewport(0, 0, frameWidth, frameHeight);
    gl.scissor(0, 0, frameWidth, frameHeight);    
    gl.useProgram(this.resizeProgram_);
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, this.texture2_);
    gl.uniform1i(this.resizeProgramInputSampler_, 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.finish();
  
    // Create a video frame from canvas and enqueue it to controller
    // alpha: 'discard' is needed in order to send frames to a PeerConnection.
    frame.close();
    controller.enqueue(new VideoFrame(this.canvas_, {timestamp: frame.timestamp, alpha: 'discard'}));
  }

  /** @override */
  destroy() {
    if (this.gl_) {
      console.log('[WebGLBackgroundBlur] Forcing WebGL context to be lost.');
      /** @type {!WEBGL_lose_context} */ (
        this.gl_.getExtension('WEBGL_lose_context'))
          .loseContext();
    }
  }
}
