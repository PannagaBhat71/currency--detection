# currency--dtetction
using roboflow i labelled different images of indian currency and trained it to yolov8 using colab
Hello everyone my name is pannaga

my pc specs are
cpu: pentium e5300
ram: 4gb ddr3
os: zorin 18.1

i will share my journey of how did i create this project 

STEP 1
shoot videos of different angles of currency notes
and create a account in roboflow and create project move the videos and it will split the frames. Set it to 2 frames per second

STEP2 LABELLING
start labelling the images using its auto detect feature 

STEP3 create versions and export as code
select the sidebar option and create a version and enable show code and copy the python code

STEP4 open colab and train

STEP5 OPTIMISATION
if ur cpu is old as mine it doesnt support avx so download onnxruntime 1.17.0
and convert ur model.pt to model.onnx

STEP 6
see the code and deploy


IMP NOTE
CONS OF MY PROJECT
I FORGOT TO TRAIN IT WITH 10 RUPEES
BCKSIDE OF THE NOTE SOMETIMES WORK AND SOMETIMES NOT

