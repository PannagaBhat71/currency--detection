
# currency detection

A brief description of what this project does and who it's for




This project does object detection of. Indian currency notes like 20,50,500,100,200. i trained yolov8. By 427 images labelled using roboflow 
in google colab

The netbook link is: https://colab.research.google.com/drive/1JK0CM_cM-v0O080YREeJ5E_rStDDtUFJ?usp=sharing





## Features

- 77 to 97% accuracy
- optimised for pentium cpu
- trained with 50 EPOCHS




## Tech Stack

**training-modules-in-python** ultralytic, roboflow, cv2, yolo

**locally-running-modules-for pentium:** onnxruntime, cv2, numpy, openvino




## Preview

![App Screenshot](https://github.com/PannagaBhat71/currency--detection/blob/main/Screenshot%20from%202026-05-24%2021-30-34.png)


## Cons

1. This model sometimes detect backside of the note and sometimes not. accuracy is lower when shown backside of the note
2. Important one i didnt train the model older notes of 20,100 and i forgot to train 10
## Support

For support, email pannagabhat886@gmail.com


## Lessons Learned

well im a beginner who just started learning opencv and object detection. It is actually easy to train a model 

But it is hard to deploy locally if ur system doesnt support avx. Like mine which uses pentium e5300

## To fix that

 1. Download Intel's official signing key
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

 2. Add the key safely to your Zorin system keyring
sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

3. link that

echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list

4. Install openvino globally

sudo apt update

sudo apt install openvino -y

5.Install onnxruntime

pip3 install onnxruntime==1.17.0 numpy --break-system-packages




## License

[MIT](https://choosealicense.com/licenses/mit/)

