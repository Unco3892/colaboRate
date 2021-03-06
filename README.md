# RstudioServer on Google Colaboratory
This repository builds a Rstudio server executed on Google colaboratory.

The goal is to build tools that make research easy for Rstudio users who do not have access to adquate hardware for their projects. Many thanks to [naru-T/RstudioServer_on_Colab](https://github.com/naru-T/RstudioServer_on_Colab) for having built the initial VM which we expand upon.


# How to launch Rstudio-server on Google Colab?
1. Open run_server.ipynb on Colab
2. Run all the chunks of code. 
3. The "rstudio" user is created. The password should be defined by user.
4. Launch the localtunnel produced weblink. Rstudio-server will be launched.

# Note
The rstudio server does not accept the root user.  
To access google drive, the "googledrive" package should be installed.  


# References
- https://towardsdatascience.com/colab-free-gpu-ssh-visual-studio-code-server-36fe1d3c5243
- https://memo.chezo.uno/Google-Colaboratory-VS-Code-code-server-3b0f4ae8181c49ecac0c99f6e4017133
