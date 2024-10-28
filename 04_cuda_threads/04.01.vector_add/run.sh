#!bin/bash

# crate the build directory if not exists

if [ ! -d "build" ]; then
    mkdir build
fi

# go to the build directory and run cmake
cd build

# run cmake to creat the makefile
cmake ..

# compile using the generated makefile
make 

# run the executable
./vectorAdd