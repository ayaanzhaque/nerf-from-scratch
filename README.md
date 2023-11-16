# NeRF from scratch (CS180 Project)

I trained a NeRF entirely from scratch. The data can be found [here](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/assets/lego_200x200.npz). [Here](https://inst.eecs.berkeley.edu/~cs180/fa23/hw/proj5/) was the spec for the project. 

Here's the result:

<div style="display: flex; justify-content: space-between;">
    <div style="flex: 0 0 calc(50% - 5px); padding: 5px;">
      <img src="assets/rgb.gif" alt="Image 2" style="max-width: 100%; height: auto;">
      <p>RGB</p>
    </div>
    <div style="flex: 0 0 calc(50% - 5px); padding: 5px;">
      <img src="assets/depth.gif" alt="Image 2" style="max-width: 100%; height: auto;">
      <p>Depth</p>
    </div>
</div>

To install and run the code:

```
conda create -n nerf -y python=3.10
conda activate nerf
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 torchdata==0.6.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

To train the nerf:

```
python nerf.py
```
