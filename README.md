# Implementing [neural style code by Justin Johnson](https://github.com/jcjohnson/neural-style) on Ubuntu 18.04 and CUDA 10.1
While implementing this code on Ubuntu 18.04 and CUDA 10.1, I faced several obstacles mainly with the installation of 
prerequisite libraries such as torch7, loadcaffe, cutorch and cunn as these libraries are not directly compatible with
the above mentioned Ubuntu and CUDA versions. 

I hope this repository would help others who are trying to install these libraries on their system and will help them
smoothly run this code.  

This is a torch implementation of the paper [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

The paper presents an algorithm for combining the content of one image with the style of another image using
convolutional neural networks. Here's an example that maps the artistic style of
[The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night)
onto a night-time photograph of the Stanford campus:

<div align="center">
 <img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night_google.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/hoovertowernight.jpg" height="223px">
 <img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/starry_stanford_bigger.png" width="710px">
</div>

Applying the style of different images to the same content image gives interesting results.
Here we reproduce Figure 2 from the paper, which renders a photograph of the Tubingen in Germany in a
variety of styles:

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/tubingen.jpg" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_shipwreck.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_seated_nude.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_composition_vii.png" height="250px">
</div>

Here are the results of applying the style of various pieces of artwork to this photograph of the
golden gate bridge:


<div align="center"
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/golden_gate.jpg" height="200px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/frida_kahlo.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_kahlo.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/escher_sphere.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_escher.png" height="160px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/woman-with-hat-matisse.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_matisse.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/the_scream.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_scream.png" height="160px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/starry_night_crop.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry.png" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/seated-nude.jpg" height="160px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_seated.png" height="160px">
</div>

### Content / Style Tradeoff

The algorithm allows the user to trade-off the relative weight of the style and content reconstruction terms,
as shown in this example where we port the style of [Picasso's 1907 self-portrait](http://www.wikiart.org/en/pablo-picasso/self-portrait-1907) onto Brad Pitt:

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/picasso_selfport1907.jpg" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/inputs/brad_pitt.jpg" height="220px">
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_5_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_1_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_01_style_10.png" height="220px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/pitt_picasso_content_0025_style_10.png" height="220px">
</div>

### Style Scale

By resizing the style image before extracting style features, we can control the types of artistic
features that are transfered from the style image; you can control this behavior with the `-style_scale` flag.
Below we see three examples of rendering the Golden Gate Bridge in the style of The Starry Night.
From left to right, `-style_scale` is 2.0, 1.0, and 0.5.

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale2.png" height=175px>
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale1.png" height=175px>
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scale05.png" height=175px>
</div>

### Multiple Style Images
You can use more than one style image to blend multiple artistic styles.

Clockwise from upper left: "The Starry Night" + "The Scream", "The Scream" + "Composition VII",
"Seated Nude" + "Composition VII", and "Seated Nude" + "The Starry Night"

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry_scream.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream_composition_vii.png" height="250px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry_seated.png" height="250px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_seated_nude_composition_vii.png" height="250px">
</div>


### Style Interpolation
When using multiple style images, you can control the degree to which they are blended:

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_3_7.png" height="175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_5_5.png" height="175px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/golden_gate_starry_scream_7_3.png" height="175px">
</div>


### Transfer style but not color
If you add the flag `-original_colors 1` then the output image will retain the colors of the original image;
this is similar to [the recent blog post by deepart.io](http://blog.deepart.io/2016/06/04/color-independent-style-transfer/).

<div align="center">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_starry.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_scream.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/tubingen_composition_vii.png" height="185px">

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/original_color/tubingen_starry.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/original_color/tubingen_scream.png" height="185px">
<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/original_color/tubingen_composition_vii.png" height="185px">
</div>

## Setup:

This guide will walk you through the setup for `neural-style` on Ubuntu 18.04 and CUDA 10.1.

### Step-1: Install CUDA

First download [CUDA 10.1](https://developer.nvidia.com/cuda-downloads) corresponding system and unpack it. 
Now update the repository cache and install CUDA. Note that this will also install a graphics driver from NVIDIA.
```
sudo apt-get update
sudo apt-get install cuda
```
At this point you may need to reboot your machine to load the new graphics driver. After rebooting, you should be able to see the status of your graphics card(s) by running the command nvidia-smi; it should give output that looks something like this:
```
Mon Sep  9 22:27:28 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  On   | 00000000:01:00.0  On |                  N/A |
| 35%   38C    P0    53W / 260W |   3919MiB / 10986MiB |     35%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 208...  On   | 00000000:03:00.0 Off |                  N/A |
| 35%   30C    P8    20W / 260W |      1MiB / 10989MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1763      G   /usr/lib/xorg/Xorg                            40MiB |
|    0      1817      G   /usr/bin/gnome-shell                          58MiB |
|    0      4995      G   /usr/lib/xorg/Xorg                          1233MiB |
|    0      5143      G   /usr/bin/gnome-shell                         738MiB |
|    0     10018      G   ...quest-channel-token=7976773505696378946  1489MiB |
|    0     11403      G   ...uest-channel-token=17206146747911347054   356MiB |
+-----------------------------------------------------------------------------+

```

### Step-2 : Install torch7

First we need to install torch.
Building Torch with CUDA10 is a troublesome work. Official [torch distro](https://github.com/torch/distro.git) will not work. We will be using [nagadomi's](https://github.com/nagadomi) modified [torch's distro](https://github.com/nagadomi/distro.git) for CUDA10. Please note that this only works with CUDA10.
```
# in a terminal, run the commands
cd ~/
git clone https://github.com/nagadomi/distro.git ~/torch --recursive
cd ~/torch
bash install-deps 	# installs all the dependences for torch. 
./install.sh 						# install lua and torch 
./update.sh								
```
`./install.sh` and also edits .bashrc to add torch to your PATH variable. Hence we source .bashrc to refresh your 
environment variable.
```
source ~/.bashrc
```
To check that your torch installation is working, run the command `th` to enter the interactive shell. To quit just type `exit`.

### Step-3 : Install loadcaffe

`loadcaffe` depends on [Google's Protocol Buffer library](https://developers.google.com/protocol-buffers/?hl=en) so we'll need to install that first:
```
sudo apt-get install libprotobuf-dev protobuf-compiler
```
Now we can instal `loadcaffe`:
```
luarocks install loadcaffe
```
### Step-4: Install neural-style

First we clone `neural-style` from GitHub:
```
cd ~/
git clone https://github.com/DhruvGamdha/Implementing-neural-style-on-Ubuntu-18.04-and-Cuda-10.1.git ~/neural-style --recursive
cd neural-style
```
Next we need to download the pretrained neural network models:
```
sh models/download_models.sh
```
This will download the original [VGG-19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md).
Leon Gatys has graciously provided the modified version of the VGG-19 model that was used in their paper;
this will also be downloaded. By default the original VGG-19 model is used.

If you have a smaller memory GPU then using NIN Imagenet model will be better and gives slightly worse yet comparable results. You can get the details on the model from [BVLC Caffe ModelZoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) and can download the files from [NIN-Imagenet Download Link](https://drive.google.com/folderview?id=0B0IedYUunOQINEFtUi1QNWVhVVU&usp=drive_web)

You should now be able to run `neural-style` in CPU mode like this:
```
th neural_style.lua -gpu -1 -print_iter 1
```
If everything is working properly you should see output like this:
```
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:505] Reading dangerously large protocol message.  If the message turns out to be larger than 1073741824 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 574671192
Successfully loaded models/VGG_ILSVRC_19_layers.caffemodel
conv1_1: 64 3 3 3
conv1_2: 64 64 3 3
conv2_1: 128 64 3 3
conv2_2: 128 128 3 3
conv3_1: 256 128 3 3
conv3_2: 256 256 3 3
conv3_3: 256 256 3 3
conv3_4: 256 256 3 3
conv4_1: 512 256 3 3
conv4_2: 512 512 3 3
conv4_3: 512 512 3 3
conv4_4: 512 512 3 3
conv5_1: 512 512 3 3
conv5_2: 512 512 3 3
conv5_3: 512 512 3 3
conv5_4: 512 512 3 3
fc6: 1 1 25088 4096
fc7: 1 1 4096 4096
fc8: 1 1 4096 1000
WARNING: Skipping content loss	
Iteration 1 / 1000	
  Content 1 loss: 2091178.593750	
  Style 1 loss: 30021.292114	
  Style 2 loss: 700349.560547	
  Style 3 loss: 153033.203125	
  Style 4 loss: 12404635.156250	
  Style 5 loss: 656.860304	
  Total loss: 15379874.666090	
Iteration 2 / 1000	
  Content 1 loss: 2091177.343750	
  Style 1 loss: 30021.292114	
  Style 2 loss: 700349.560547	
  Style 3 loss: 153033.203125	
  Style 4 loss: 12404633.593750	
  Style 5 loss: 656.860304	
  Total loss: 15379871.853590	
```
### Step-5: Install CUDA backend for torch
`luarocks install cutorch` and `luarocks install cunn` will not work because it downloads cutorch and cunn from original cutorch and cunn versions from internet which will not work in our case due to incompatibility issues. To install compatible cutorch and cunn from local disk, use the following commands.
```
cd ~/torch/extra/cutorch
luarocks make rocks/cutorch-scm-1.rockspec
cd ~/torch/extra/cunn
luarocks make rocks/cunn-scm-1.rockspec
```
You can check that the installation worked by running the following:
```
th -e "require 'cutorch'; require 'cunn'; print(cutorch)"
```
This should produce output like the this:
```
{
  getStream : function: 0x40d40ce8
  getDeviceCount : function: 0x40d413d8
  setHeapTracking : function: 0x40d41a78
  setRNGState : function: 0x40d41a00
  getBlasHandle : function: 0x40d40ae0
  reserveBlasHandles : function: 0x40d40980
  setDefaultStream : function: 0x40d40f08
  getMemoryUsage : function: 0x40d41480
  getNumStreams : function: 0x40d40c48
  manualSeed : function: 0x40d41960
  synchronize : function: 0x40d40ee0
  reserveStreams : function: 0x40d40bf8
  getDevice : function: 0x40d415b8
  seed : function: 0x40d414d0
  deviceReset : function: 0x40d41608
  streamWaitFor : function: 0x40d40a00
  withDevice : function: 0x40d41630
  initialSeed : function: 0x40d41938
  CudaHostAllocator : torch.Allocator
  test : function: 0x40ce5368
  getState : function: 0x40d41a50
  streamBarrier : function: 0x40d40b58
  setStream : function: 0x40d40c98
  streamBarrierMultiDevice : function: 0x40d41538
  streamWaitForMultiDevice : function: 0x40d40b08
  createCudaHostTensor : function: 0x40d41670
  setBlasHandle : function: 0x40d40a90
  streamSynchronize : function: 0x40d41590
  seedAll : function: 0x40d414f8
  setDevice : function: 0x40d414a8
  getNumBlasHandles : function: 0x40d409d8
  getDeviceProperties : function: 0x40d41430
  getRNGState : function: 0x40d419d8
  manualSeedAll : function: 0x40d419b0
  _state : userdata: 0x022fe750
}
```
You should now be able to run `neural-style` in GPU mode:
```
th neural_style.lua -gpu 0 -print_iter 1
```
### Step 6: Install cuDNN

cuDNN is a library from NVIDIA that efficiently implements many of the operations (like convolutions and pooling) that are commonly used in deep learning.
After registering as a developer with NVIDIA, you can [download cuDNN here](https://developer.nvidia.com/cudnn). Download Version 7.6 for CUDA 10.1 and Ubuntu 18.04.
After dowloading and unpacking, copy and paste cuDNN files like this:
```
# open directory containing the unpacked directory and run the following commands.
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.1/include/
```
Next we need to install the torch bindings for cuDNN, `luarocks install cudnn` will not work. The master branch of cuDNN.torch does not support cuDNN v7. Installing from R7 branch probably works fine.
```
cd ~/
git clone https://github.com/soumith/cudnn.torch.git -b R7
cd cudnn.torch
luarocks make cudnn-scm-1.rockspec
```
You should now be able to run `neural-style` with cuDNN like this:
```
th neural_style.lua -gpu 0 -backend cudnn
```
Note that the cuDNN backend can only be used for GPU mode.

## Usage
Basic usage:
```
th neural_style.lua -style_image <image.jpg> -content_image <image.jpg>
```

OpenCL usage with NIN Model (This requires you download the NIN Imagenet model files as described above):
```
th neural_style.lua -style_image examples/inputs/picasso_selfport1907.jpg -content_image examples/inputs/brad_pitt.jpg -output_image profile.png -model_file models/nin_imagenet_conv.caffemodel -proto_file models/train_val.prototxt -gpu 0 -backend clnn -num_iterations 1000 -seed 123 -content_layers relu0,relu3,relu7,relu12 -style_layers relu0,relu3,relu7,relu12 -content_weight 10 -style_weight 1000 -image_size 512 -optimizer adam
```

![OpenCL NIN Model Picasso Brad Pitt](/examples/outputs/pitt_picasso_nin_opencl.png)


To use multiple style images, pass a comma-separated list like this:

`-style_image starry_night.jpg,the_scream.jpg`.

Note that paths to images should not contain the `~` character to represent your home directory; you should instead use a relative
path or a full absolute path.

**Options**:
* `-image_size`: Maximum side length (in pixels) of of the generated image. Default is 512.
* `-style_blend_weights`: The weight for blending the style of multiple style images, as a
  comma-separated list, such as `-style_blend_weights 3,7`. By default all style images
  are equally weighted.
* `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `-gpu` to -1.

**Optimization options**:
* `-content_weight`: How much to weight the content reconstruction term. Default is 5e0.
* `-style_weight`: How much to weight the style reconstruction term. Default is 1e2.
* `-tv_weight`: Weight of total-variation (TV) regularization; this helps to smooth the image.
  Default is 1e-3. Set to 0 to disable TV regularization.
* `-num_iterations`: Default is 1000.
* `-init`: Method for generating the generated image; one of `random` or `image`.
  Default is `random` which uses a noise initialization as in the paper; `image`
  initializes with the content image.
* `-optimizer`: The optimization algorithm to use; either `lbfgs` or `adam`; default is `lbfgs`.
  L-BFGS tends to give better results, but uses more memory. Switching to ADAM will reduce memory usage;
  when using ADAM you will probably need to play with other parameters to get good results, especially
  the style weight, content weight, and learning rate; you may also want to normalize gradients when
  using ADAM.
* `-learning_rate`: Learning rate to use with the ADAM optimizer. Default is 1e1.
* `-normalize_gradients`: If this flag is present, style and content gradients from each layer will be
  L1 normalized. Idea from [andersbll/neural_artistic_style](https://github.com/andersbll/neural_artistic_style).

**Output options**:
* `-output_image`: Name of the output image. Default is `out.png`.
* `-print_iter`: Print progress every `print_iter` iterations. Set to 0 to disable printing.
* `-save_iter`: Save the image every `save_iter` iterations. Set to 0 to disable saving intermediate results.

**Layer options**:
* `-content_layers`: Comma-separated list of layer names to use for content reconstruction.
  Default is `relu4_2`.
* `-style_layers`: Comma-separated list of layer names to use for style reconstruction.
  Default is `relu1_1,relu2_1,relu3_1,relu4_1,relu5_1`.

**Other options**:
* `-style_scale`: Scale at which to extract features from the style image. Default is 1.0.
* `-original_colors`: If you set this to 1, then the output image will keep the colors of the content image.
* `-proto_file`: Path to the `deploy.txt` file for the VGG Caffe model.
* `-model_file`: Path to the `.caffemodel` file for the VGG Caffe model.
  Default is the original VGG-19 model; you can also try the normalized VGG-19 model used in the paper.
* `-pooling`: The type of pooling layers to use; one of `max` or `avg`. Default is `max`.
  The VGG-19 models uses max pooling layers, but the paper mentions that replacing these layers with average
  pooling layers can improve the results. I haven't been able to get good results using average pooling, but
  the option is here.
* `-backend`: `nn`, `cudnn`, or `clnn`. Default is `nn`. `cudnn` requires
  [cudnn.torch](https://github.com/soumith/cudnn.torch) and may reduce memory usage.
  `clnn` requires [cltorch](https://github.com/hughperkins/cltorch) and [clnn](https://github.com/hughperkins/clnn)
* `-cudnn_autotune`: When using the cuDNN backend, pass this flag to use the built-in cuDNN autotuner to select
  the best convolution algorithms for your architecture. This will make the first iteration a bit slower and can
  take a bit more memory, but may significantly speed up the cuDNN backend.

## Frequently Asked Questions

**Problem:** Generated image has saturation artifacts:

<img src="https://cloud.githubusercontent.com/assets/1310570/9694690/fa8e8782-5328-11e5-9c91-11f7b215ad19.png">

**Solution:** Update the `image` packge to the latest version: `luarocks install image`

**Problem:** Running without a GPU gives an error message complaining about `cutorch` not found

**Solution:**
Pass the flag `-gpu -1` when running in CPU-only mode

**Problem:** The program runs out of memory and dies

**Solution:** Try reducing the image size: `-image_size 256` (or lower). Note that different image sizes will likely
require non-default values for `-style_weight` and `-content_weight` for optimal results.
If you are running on a GPU, you can also try running with `-backend cudnn` to reduce memory usage.

**Problem:** Get the following error message:

`models/VGG_ILSVRC_19_layers_deploy.prototxt.cpu.lua:7: attempt to call method 'ceil' (a nil value)`

**Solution:** Update `nn` package to the latest version: `luarocks install nn`

**Problem:** Get an error message complaining about `paths.extname`

**Solution:** Update `torch.paths` package to the latest version: `luarocks install paths`

**Problem:** NIN Imagenet model is not giving good results. 

**Solution:** Make sure the correct `-proto_file` is selected. Also make sure the correct parameters for `-content_layers` and `-style_layers` are set. (See OpenCL usage example above.)

**Problem:** `-backend cudnn` is slower than default NN backend

**Solution:** Add the flag `-cudnn_autotune`; this will use the built-in cuDNN autotuner to select the best convolution algorithms.

## Memory Usage
By default, `neural-style` uses the `nn` backend for convolutions and L-BFGS for optimization.
These give good results, but can both use a lot of memory. You can reduce memory usage with the following:

* **Use cuDNN**: Add the flag `-backend cudnn` to use the cuDNN backend. This will only work in GPU mode.
* **Use ADAM**: Add the flag `-optimizer adam` to use ADAM instead of L-BFGS. This should significantly
  reduce memory usage, but may require tuning of other parameters for good results; in particular you should
  play with the learning rate, content weight, style weight, and also consider using gradient normalization.
  This should work in both CPU and GPU modes.
* **Reduce image size**: If the above tricks are not enough, you can reduce the size of the generated image;
  pass the flag `-image_size 256` to generate an image at half the default size.
  
With the default settings, `neural-style` uses about 3.5GB of GPU memory on my system;
switching to ADAM and cuDNN reduces the GPU memory footprint to about 1GB.

## Speed
Speed can vary a lot depending on the backend and the optimizer.
Here are some times for running 500 iterations with `-image_size=512` on a Maxwell Titan X with different settings:
* `-backend nn -optimizer lbfgs`: 62 seconds
* `-backend nn -optimizer adam`: 49 seconds
* `-backend cudnn -optimizer lbfgs`: 79 seconds
* `-backend cudnn -cudnn_autotune -optimizer lbfgs`: 58 seconds
* `-backend cudnn -cudnn_autotune -optimizer adam`: 44 seconds
* `-backend clnn -optimizer lbfgs`: 169 seconds
* `-backend clnn -optimizer adam`: 106 seconds 

Here are the same benchmarks on a Pascal Titan X with cuDNN 5.0 on CUDA 8.0 RC:
* `-backend nn -optimizer lbfgs`: 43 seconds
* `-backend nn -optimizer adam`: 36 seconds
* `-backend cudnn -optimizer lbfgs`: 45 seconds
* `-backend cudnn -cudnn_autotune -optimizer lbfgs`: 30 seconds
* `-backend cudnn -cudnn_autotune -optimizer adam`: 22 seconds

## Multi-GPU scaling
You can use multiple GPUs to process images at higher resolutions; different layers of the network will be
computed on different GPUs. You can control which GPUs are used with the `-gpu` flag, and you can control
how to split layers across GPUs using the `-multigpu_strategy` flag.

For example in a server with four GPUs, you can give the flag `-gpu 0,1,2,3` to process on GPUs 0, 1, 2, and
3 in that order; by also giving the flag `-multigpu_strategy 3,6,12` you indicate that the first two layers
should be computed on GPU 0, layers 3 to 5 should be computed on GPU 1, layers 6 to 11 should be computed on
GPU 2, and the remaining layers should be computed on GPU 3. You will need to tune the `-multigpu_strategy`
for your setup in order to achieve maximal resolution.

We can achieve very high quality results at high resolution by combining multi-GPU processing with multiscale
generation as described in the paper
<a href="https://arxiv.org/abs/1611.07865">**Controlling Perceptual Factors in Neural Style Transfer**</a> by Leon A. Gatys, 
Alexander S. Ecker, Matthias Bethge, Aaron Hertzmann and Eli Shechtman.

Here is a 3620 x 1905 image generated on a server with four Pascal Titan X GPUs:

<img src="https://raw.githubusercontent.com/jcjohnson/neural-style/master/examples/outputs/starry_stanford_bigger.png" height="400px">

The script used to generate this image <a href='examples/multigpu_scripts/starry_stanford.sh'>can be found here</a>.

## Implementation details
Images are initialized with white noise and optimized using L-BFGS.

We perform style reconstructions using the `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, and `conv5_1` layers
and content reconstructions using the `conv4_2` layer. As in the paper, the five style reconstruction losses have
equal weights.

## Citation

If you find this code useful for your research, please cite:

```
@misc{Johnson2015,
  author = {Johnson, Justin},
  title = {neural-style},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jcjohnson/neural-style}},
}
```
