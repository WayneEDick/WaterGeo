@echo off

mkdir pack
rem --- copy the 10 diagnostic files ---

copy /y "G2b_build_ccs.json" pack
copy /y "G3_classify_ccs.json" pack
copy /y "G4_page_constants.json" pack
copy /y "P2a_V_Graph.json" pack
copy /y "P2a_dbg_VG_boxes_only.json" pack
copy "C:\Users\wayne\OneDrive\Documents\Jupyter\water\Geo\Code\input\input.png" pack
copy /y "input_g3_boxes_only.png" pack
copy /y "input_vg_boxes_only.png" pack
copy /y "input_hg_boxes_only_CSUN.png" pack
copy /y "input_g4_boxes.png" pack

pause