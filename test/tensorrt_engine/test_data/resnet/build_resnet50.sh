trtexec --onnx=ResNet50.onnx --saveEngine=ResNet50.engine --minShapes=data:1x3x224x224 --optShapes=data:2x3x224x224 --maxShapes=data:4x3x224x224 --fp16
