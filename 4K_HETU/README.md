###### **Overview**
The code files in this directory are part of the ESCIM hardware testing system's upper computer software SDK, which requires linking to the ESCIM hardware platform to support operation. The ESCIM hardware system is designed for testing a 4K ReRAM array (128Row x 32Column). This system consists of three parts: the upper computer program and SDK on the PC side, the ZC706 FPGA development board, and the 4K chip testing board.

The control program on the PC side is used for controlling the testing system and processing the result data; the FPGA serves as the bridge of the system, on one hand, it receives instructions from the control program, controls the testing board, and performs operations on the chips under test, on the other hand, it reads the results returned by the testing board according to the timing requirements, and sends them back to the control program; the chip testing board carries the chips under test, and provides the necessary power, signal generation circuits, and chip output signal processing circuits for the chips under test.

###### **INSTALLATION GUIDE**

Installation not required.

###### **Main Files description**
1. To run upper computer program using UI：
	>python .\app\main.py
2. To develop user-defined software using SDK, please import:
	'''
	import apps
	import arrayexp
	import arrayparams
	import reram_register
	'''
