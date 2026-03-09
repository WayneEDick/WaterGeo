@echo off

del pack\*

copy /y "G4_page_constants.json" pack
copy /y "P2a_V_Graph.json" pack
copy /y "P2a_dbg_VG_boxes_only.json" pack
copy /y "P2c_H_Graph.json" pack
copy /y "P2c_H_Graph.json" pack
copy /y "P2c_dbg_HG_boxes_only.json" pack
copy /y "input_g5a_tokens.yaml" pack

pause