 "PBTA: Partial Break Triplet Attention Model for Small Pedestrian Detection Based on Vehicle Camera Sensors"

Code overview:
	Successfully detecting small pedestrians through vehicle camera sensors would greatly facilitate the development of autonomous driving safety applications. However, the existing pedestrian detection models applied to vehicle camera applications were limited by the scale confusion problem and the weak feature problem of small pedestrian targets. To resolve these issues, this study proposed a PBTA network composed of two components: the Partial Break Bidirectional Feature Pyramid Network (PBFPN) and the TR-NCSPDarknet53. PBFPN was used to solve the scale confusion problem in shallow feature maps by employing a partial break operation and a branch fusion operation. In TR-NCSPDarknet53, Ta-conv module was proposed to solve the weak feature problem. The PBTA network provided a new improvement idea for small pedestrian detection. It greatly improved the accuracy while keeping the parameters at a low level, which is vital for safety applications in autonomous driving. Extensive experiments on the CityPersons dataset, which includes various traffic images from camera sensors, demonstrates the accuracy of the PBTA network in small pedestrian detection. 

Installation description:
1. Install the corresponding library through the requirements file
2. Use setup.py to install the project in the environment
3. Use the corresponding command line for training and testing

Validated datasets:
Crowdhuman、Widerperson、Citypersons.
Experimental results refer to the content of the paper
Note: Pre-trained weights were not used to participate in the experiment.

Code structure:
Regarding the structural code reference of PBFPN and PBTA: PBTA-result-code\PBTA\ultralytics\models\PBTA-PBFPN-Ta-conv
Regarding the structural code reference of Ta-conv: C:\Users\11366\Desktop\PBTA-result-code\PBTA\ultralytics\nn\modules.py

Author's contact information: g1136639260@163.com

Acknowledgements:
Thanks to all people or organizations who contributed to the project.