@echo off

del pack\*.*
mkdir pack

copy "C:\Users\wayne\OneDrive\Documents\Jupyter\water\Geo\Code\input\input.png" pack
copy /y "input_hg_boxes_only_CSUN.png" pack
copy /y "G5a_classify_tokens.json" pack
copy /y "P2c_H_Graph.json" pack
copy /y "P2a_V_Graph.json" pack
copy /y "" pack

pause