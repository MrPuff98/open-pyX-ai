# open-pyX-ai
Welcome to open-pyX-ai, the machine learning-based software for the accurate deconvolution of X-ray diffraction and total X-ray scattering data

To proceed, simply run one of the following scripts:

##########################################
    'Classic' algorithm implementation:
    pyLearn - evaluates the instrumental broadening function 
        input:  
            'perfect' XRD pattern without instrumental broadening
            experimental ('blurred') XRD pattern
        output:
            .pth model weights file - instrumental blurring function for pyCore.XRDConv
    
    pyRLCustom - applies the modified Richardson-Lucy deconvolution algorithm to restore the blurred XRD pattern
        input: 
            experimental ('blurred') XRD pattern
            instrumental blurring function for pyCore.XRDConv: .pth file
        output:
            restored (deconvolved) XRD patterns
    
##########################################
    Deep learning-based approach implementation:
    genXRD_mode - generates synthetic XRD patterns for further use to train DecCNN
    
    
    pyNNLearn - trains and validates DecCNN on the basis of specified synthetic data sets

    
    pyNNDeconv.py - deconvolves the blurred XRD pattern using pre-trained DecCNN
        input: 
            experimental ('blurred') XRD pattern
            instrumental blurring function for pyCore.XRDConv: .pth file
        output:
            restored (deconvolved) XRD patterns
