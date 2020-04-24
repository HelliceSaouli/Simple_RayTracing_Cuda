################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../source/Kernels.cu 

CPP_SRCS += \
../source/Camera.cpp \
../source/Display.cpp \
../source/Fps.cpp \
../source/Mesh.cpp \
../source/PixelBuffer.cpp \
../source/Shader.cpp \
../source/TextureGL.cpp \
../source/Transform.cpp \
../source/input.cpp \
../source/main.cpp 

OBJS += \
./source/Camera.o \
./source/Display.o \
./source/Fps.o \
./source/Kernels.o \
./source/Mesh.o \
./source/PixelBuffer.o \
./source/Shader.o \
./source/TextureGL.o \
./source/Transform.o \
./source/input.o \
./source/main.o 

CU_DEPS += \
./source/Kernels.d 

CPP_DEPS += \
./source/Camera.d \
./source/Display.d \
./source/Fps.d \
./source/Mesh.d \
./source/PixelBuffer.d \
./source/Shader.d \
./source/TextureGL.d \
./source/Transform.d \
./source/input.d \
./source/main.d 


# Each subdirectory must supply rules for building sources it contributes
source/%.o: ../source/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_30,code=sm_30 -m64 -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -std=c++11 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

source/%.o: ../source/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -std=c++11 -gencode arch=compute_30,code=sm_30 -m64 -odir "source" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.2/bin/nvcc -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


