FROM quay.io/fenicsproject/stable:2019.1.0.r3
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install gmsh libgl1-mesa-glx xvfb -y
WORKDIR /APP/INSTALLS
COPY ./fenics_shells /app/installs/fenics_shells
WORKDIR /app/installs/fenics_shells
RUN python3 setup.py install
#RUN pip3 install --upgrade matplotlib numpy scipy
RUN pip3 install ipython
WORKDIR /APP
RUN pip3 install -U ipywidgets
RUN pip3 install pyvista gmsh vedo ipyvtklink
ENV SHELL=/bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]